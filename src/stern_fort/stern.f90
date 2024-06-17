! Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)
!
! Copyright © 2023, United States Government, as represented by the Administrator
! of the National Aeronautics and Space Administration. All rights reserved.
!
! The PySA, a powerful tool for solving optimization problems is licensed under
! the Apache License, Version 2.0 (the "License"); you may not use this file
! except in compliance with the License. You may obtain a copy of the License at
! http://www.apache.org/licenses/LICENSE-2.0.
!
! Unless required by applicable law or agreed to in writing, software distributed
! under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
! CONDITIONS OF ANY KIND, either express or implied. See the License for the
! specific language governing permissions and limitations under the License.

module stern
  use omp_lib
  use lincode
  implicit none
  ! integer(1), dimension(:,:), allocatable :: Ginp
  ! Rendundancy column
  integer(1), parameter :: rdnbit = 0
  integer(1), parameter :: rdnty = ibset(0, rdnbit)
  ! Infomation set column
  integer(1), parameter :: isdbit = 1
  integer(1), parameter :: isdty = ibset(0, isdbit)
  ! Permanent column
  integer(1), parameter :: permbit = 2
  integer(1), parameter :: permty = ibset(0, permbit)
  ! Auxiliary column
  integer(1), parameter :: auxbit = 3
  integer(1), parameter :: auxty = ibset(0, auxbit)

  integer(4), parameter :: L0 = 8
  type SternAtk
    ! For an [n, k, d] code, nvars=n and ncode=k
    integer :: nvars
    integer :: ncode
    integer :: nclauses
    integer :: naux
    ! Parity check matrix [n-k, n] or generator matrix [k, n]
    integer(1), dimension(:, :), allocatable :: G
    ! Chosen redundancy columns (n-k)
    integer, dimension(:), allocatable :: rdncols
    ! Information set columns (k)
    ! integer, dimension(:), allocatable :: isdcols
    ! Type of each column
    ! 0 - unassigned
    ! 1 - redundancy
    ! 2 - information set
    ! 4 - permanent
    ! 8 - auxiliary
    integer(1), dimension(:), allocatable :: coltypes
    ! Work arrays
    ! Array of length ncode
    integer, dimension(:), allocatable :: workn
    ! Arrays of length k
    integer, dimension(:), allocatable :: workisd
  contains
    procedure :: sternatk_constructor
    procedure :: sternatk_gauselm
    procedure :: sternatk_destructor
    procedure :: sternatk_restart_rdn
    procedure :: sternatk_set_workisd
  end type SternAtk

  type SternComboWrk
    ! Work arrays for a combinatorial Stern attack
    integer :: nvars
    integer :: ncode
    integer :: nclauses
    integer :: l ! number of bits from the redundancy set to collide
    integer :: p ! size of linear combinations in each split
    integer :: m ! number of l-combinations to check for collisions
    logical :: qcol ! Perform "quick" collision checks
    integer, dimension(2) :: kx
    integer, dimension(2) :: nx  ! number of possible p-combinations in each split
    integer(1), dimension(:, :, :), allocatable :: combos
    integer, dimension(:, :), allocatable :: colpos ! collision positions (l, m)
    integer(1), dimension(:, :), allocatable :: colwork ! clause work array (n-k, m)
    integer, dimension(:, :), allocatable :: isdsols ! ( 2p, m )
    integer(1), dimension(:, :), allocatable :: bestsols ! Best solutions (n, m)
  contains
    procedure :: sterncombowrk_constructor
  end type SternComboWrk
contains
  subroutine stern_from_cptr(stern_atk, garr, numvars, numclauses)
    ! Initialize from a C pointer to a 2D array
    integer(4) i, j
    integer(1), intent(in) :: garr(*)
    integer(4), intent(in) :: numvars, numclauses
    class(SternAtk), intent(out) :: stern_atk

    call stern_atk%sternatk_constructor(numvars, numvars - numclauses, naux = 1)
    do i = 1, numclauses
      stern_atk%rdncols(i) = i
      stern_atk%coltypes(i) = rdnty
    end do

    do i = 1, numclauses
      do j = 1, numvars + 1
        stern_atk%G(i, j) = garr(1 + (i - 1) * (numvars + 1) + (j - 1))
      end do

    end do
    call stern_atk%sternatk_gauselm(0)
  end subroutine

  subroutine stern_from_harr(stern_atk, arr, naux)
    ! Initialize from augmented parity check array
    integer(4) i
    integer(1), intent(in) :: arr(:, :)
    integer(4), optional :: naux
    class(SternAtk), intent(out) :: stern_atk
    integer(4) naux_
    integer(4) :: numvars, numclauses

    if(.not. present(naux)) then
      naux_ = 1
    else
      naux_ = naux
    end if
    numclauses = size(arr, 1)
    numvars = size(arr, 2) - naux
    if(numvars <= 0) then
      write(*, *) "Invalid input (stern_from_harr)"
      error stop
    end if

    call stern_atk%sternatk_constructor(numvars, numvars - numclauses, naux = naux)
    do i = 1, numclauses
      stern_atk%rdncols(i) = i
      stern_atk%coltypes(i) = rdnty
    end do

    stern_atk%G(:, :) = arr(:, :)

    call stern_atk%sternatk_gauselm(0)
  end subroutine


  subroutine sternatk_constructor(this, n, k, naux)
    class(SternAtk), intent(out) :: this
    integer, intent(in) :: n, k
    integer, intent(in), optional :: naux
    integer naux_
    this%nvars = n
    this%ncode = k
    if(.not. present(naux)) then
      naux_ = 1 ! Include one additonal column
    else
      naux_ = naux
    end if
    this%naux = naux

    this%nclauses = n - k

    allocate(this%G(this%nclauses, n + naux_))
    allocate(this%rdncols(n - k))
    allocate(this%coltypes(n))
    allocate(this%workn(n))
    allocate(this%workisd(k))
  end subroutine

  subroutine sternatk_destructor(this)
    class(SternAtk), intent(inout) :: this
    if(allocated(this%G)) deallocate(this%G)
    if(allocated(this%rdncols)) deallocate(this%rdncols)
    if(allocated(this%coltypes)) deallocate(this%coltypes)
    if(allocated(this%workn)) deallocate(this%workn)
  end subroutine

  subroutine sternatk_gauselm(this, getype)
    class(SternAtk), intent(inout) :: this
    integer, intent(in) :: getype
    integer rank
    if(getype == 0) then ! Direct gaussian elimination
      call gaussian_elim(this%G(:, :), rank)
    elseif(getype == 1) then ! eliminate redundancy columns
      call gaussian_elim(this%G(:, :), rank, this%rdncols)
    end if
  end subroutine sternatk_gauselm

  subroutine sternatk_update_rdn(this, row, new_col)
    ! Replace the redundancy column corresponding to a row, if possible
    class(SternAtk), intent(inout) :: this
    integer, intent(in) :: row, new_col
    integer old_col, i
    integer(1) newcolty

    newcolty = this%coltypes(new_col)
    if(.not. (btest(newcolty, rdnbit) .or. btest(newcolty, permbit))) then
      old_col = this%rdncols(row)
      this%coltypes(old_col) = ibclr(this%coltypes(old_col), rdnbit)
      this%coltypes(new_col) = ibset(this%coltypes(new_col), rdnbit)
      this%rdncols(row) = new_col
      ! eliminate the column
      do i = 1, this%nclauses
        if(i /= row .and. this%G(i, new_col)>0) then
          this%G(i, :) = ieor(this%G(i, :), this%G(row, :))
        end if
      end do
    end if
  end subroutine sternatk_update_rdn

  subroutine sternatk_restart_rdn(this)
    ! Perform gaussian elimination on a random permutation of all of the columns
    ! and select a new, uncorrelated redundancy set.
    use randutil
    class(SternAtk), intent(inout) :: this
    integer n, i, i2, j, rank
    logical tmpl
    n = this%nvars

    do i = 1, n
      this%workn(i) = i
    end do
    call ishuffle(this%workn(1:n))
    call gaussian_elim(this%G(:, :), rank, this%workn(1:n))
    if(rank /= this%nclauses) then
      print '(A, I8, I8)', "Assertion failed (rank == nclauses), ", rank, this%nclauses
      error stop
    end if
    ! Find the pivots used for elimination and classify the IR split
    i = 1
    do i2 = 1, n
      j = this%workn(i2)
      tmpl = .false. ! avoid assuming short-circuiting
      if(i<=rank) tmpl = (this%G(i, j)>0)
      if(i<=rank .and. tmpl) then
        this%rdncols(i) = j
        this%coltypes(j) = rdnty
        i = i + 1
      else
        this%coltypes(j) = isdty
      end if
    end do

    do i2 = 1, this%nclauses
      j = this%rdncols(i2)
      if(this%G(i2, j) /= 1) then
        print '(A)', "Assertion failed."
        error stop
      end if
      do i = 1, this%nclauses
        if(i/=i2 .and. this%G(i, j) /= 0) then
          print '(A)', "Assertion failed."
          error stop
        end if
      end do
    end do
  end subroutine sternatk_restart_rdn

  subroutine sternatk_set_workisd(this)
    ! List the information set columns in the workisd array
    class(SternAtk), intent(inout) :: this
    integer i, j, k, n
    n = this%nvars
    k = this%ncode
    i = 1
    do j = 1, n
      if (.not. btest(this%coltypes(j), rdnbit)) then
        this%workisd(i) = j
        i = i + 1
      end if
    end do
    if(i /= k + 1) then
      print '(A,I8,I8)', "Assertion failed: not_rdn == isd, ", (i - 1), k
      error stop
    end if
  end subroutine


  subroutine sterncombowrk_constructor(this, n, k, l, p, m, qcol)
    use mathutil
    class(SternComboWrk), intent(inout) :: this
    integer, intent(in) :: n, k
    integer, intent(in) :: l, p, m
    logical, intent(in) :: qcol
    this%nvars = n
    this%ncode = k
    this%nclauses = n - k
    this%p = p

    ! partition sizes of the information set
    this%kx(1) = k / 2
    this%kx(2) = k - this%kx(1)
    this%nx(1) = n_C_r(int(this%kx(1), 8), int(p, 8))
    this%nx(2) = n_C_r(int(this%kx(2), 8), int(p, 8))
    allocate(this%combos(this%nclauses, max(this%nx(1), this%nx(2)), 2))

    this%qcol = qcol

    if(l>0 .and. .not. qcol) then
      allocate(this%colpos(l, m))
      this%m = m
      this%l = l
    elseif(l>=L0 .and. qcol) then
      this%l = l / L0
      this%m = m
    else ! heavy stern
      this%l = l
      this%m = 1
      this%qcol = .false.
    end if

    allocate(this%colwork(this%nclauses, this%m))
    allocate(this%isdsols(2 * p, this%m))
    allocate(this%bestsols(n, this%m))

  end subroutine sterncombowrk_constructor
end module stern

