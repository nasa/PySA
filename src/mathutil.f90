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

module mathutil
  implicit none
  type ComboIter
    ! Iterator over nCk combinations of the indices {0, 1, ..., n-1}
    integer :: n
    integer :: k
    integer, dimension(64) :: c !
  contains
    procedure :: comboiter_init
    procedure :: comboiter_next
  end type ComboIter
contains
  pure function n_C_r(n, r) result(bin)
    integer(4) :: bin
    integer(8), intent(in) :: n
    integer(8), intent(in) :: r

    integer(8) :: num
    integer(8) :: den
    integer(8) :: i
    integer(8) :: k
    integer(8), parameter :: primes(*) = [2, 3, 5, 7, 11, 13, 17, 19]
    num = 1
    den = 1
    if(n < r) then
      bin = 0
    elseif(n==r) then
      bin = 1
    else
      do i = 0, r - 1
        num = num * (n - i)
        den = den * (i + 1)
        if (i > 0) then
          ! Divide out common prime factors
          do k = 1, size(primes)
            if (mod(i, primes(k)) == 0) then
              num = num / primes(k)
              den = den / primes(k)
            end if
          end do
        end if
      end do
      bin = int(num / den, 4)
    end if
  end function n_C_r
  pure function log2(x)
    real(8), intent(in) :: x
    real(8) log2
    log2 = log10(x) / log10(2.0)
  end function
  pure function  ShannonH(p)
    real(8), intent(in) :: p
    real(8) ShannonH
    if(p <= 0.0 .or. p>=1.0) then
      ShannonH = 1.0
    else
      ShannonH = - p * log2(p) - (1.0 - p) * log2(1.0 - p)
    end if
  end function


  pure function combo_to_lin(c) result(n)
    ! Map a k-combination as a combinadic to its linear index
    integer(8), intent(in), dimension(:) :: c
    integer(8) :: n
    integer(8) k, i
    n = 0
    k = size(c, 1)
    do i = 1, k
      n = n + n_C_r(c(i), i)
    end do
  end function combo_to_lin

  pure subroutine lin_combo_res(idx, k, ak, res)
    ! Find the largest ak such that nCr(ak, k) <= idx
    ! Return ak and the residual idx - nCr(ak, k)
    integer, intent(in) :: idx, k
    integer, intent(out) :: ak, res
    integer oldres
    if(idx<=0) then
      ak = k - 1
      res = 0
    elseif(idx==1) then
      ak = k
      res = 0
    else
      ak = k + 1
      oldres = idx - 1
      do
        res = idx - n_C_r(int(ak, 8), int(k, 8))
        if(res < 0) then
          res = oldres
          ak = ak - 1
          return
        else
          oldres = res
          ak = ak + 1
        end if
      end do
    end if
  end subroutine lin_combo_res

  subroutine lin_to_combo(idx, k, c)
    ! Map a linear index to its k-combinatorial number representation
    integer, intent(in) :: idx, k
    integer, dimension(k), intent(out) :: c
    integer res, i, newres
    res = idx
    do i = k, 1, -1
      call lin_combo_res(res, i, c(i), newres)
      res = newres
    end do
  end subroutine lin_to_combo

  subroutine comboiter_init(this, n, k)
    class(ComboIter), intent(out) :: this
    integer, intent(in) :: n, k
    integer i
    if(k > 64) then
      print '(A)', "Assertion error: will not iterate over combinations more than k=64. "
      error stop
    end if
    this%n = n
    this%k = k
    do i = 1, k
      this%c(i) = i - 1
    end do
  end subroutine comboiter_init

  function comboiter_next(this) result(iter)
    class(ComboIter), intent(inout) :: this
    logical iter
    integer i, i2, k, nxt
    k = this%k
    i = 1 ! Find the first combinatorial index that can be incremented
    do i = 1, k
      if(i==k) then
        nxt = this%n + 1
      else
        nxt = this%c(i + 1)
      end if
      if(this%c(i) /= nxt - 1) then
        this%c(i) = this%c(i) + 1
        do i2 = 1, i - 1
          this%c(i2) = i2 - 1
        end do
        exit
      end if
    end do
    if(this%c(k) >= this%n) then
      iter = .false.
    else
      iter = .true.
    end if

  end function comboiter_next
end module mathutil