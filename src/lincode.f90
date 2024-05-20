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

module lincode
    implicit none
    
contains
subroutine print_f2_array(x)
    integer(1), dimension(:), intent(in) :: x
    integer(1), dimension(:), allocatable :: b
    integer(8) nbytes, nbits, i, j, k

    nbits = size(x, 1)
    nbytes = 1 + (nbits/8)
    if(mod(nbits, int(8,8)) == 0) nbytes = nbytes - 1
    allocate(b(nbytes))

    do i = 1, nbytes
        b(i)=0
        do j = 1, 8
            k=(i-1)*8 + j
            if(k>nbits) exit
            if(x(k)>0) b(i)=ibset(b(i), j-1)
        end do
    end do
    print '(*(Z3.2))', b

end subroutine

subroutine gaussian_elim(Amat, rank, cols, pivotcols)
    ! Perform binary Gaussian elimination on the matrix
    integer(1), dimension(:, :), intent(inout) :: Amat
    integer, intent(out) :: rank
    integer, dimension(:), intent(in), optional :: cols
    integer, dimension(:), intent(out), optional :: pivotcols
    integer n, m, redcols, i, j, j2, colj, rowi, pivt
    integer(1) tmp

    n = size(Amat, 1)
    m = size(Amat, 2)
    redcols = m ! maximum number of columns to try to reduce
    rowi=1
    if(present(cols)) then ! Reduce over specific columns
        redcols = size(cols, 1)
    end if
    do j=1,redcols
        if(present(cols)) then
            colj = cols(j)
        else
            colj=j
        end if
        ! Find potential pivots
        do i=rowi,n
            if(Amat(i,colj) > 0) then
                pivt=i
                goto 100
            end if
        end do
        goto 200
100         continue
        if(pivt /= rowi) then ! swap rows pivt and rowi
            do j2=1,m
                tmp = Amat(pivt, j2)
                Amat(pivt, j2) = Amat(rowi, j2)
                Amat(rowi, j2) = tmp
            end do
        end if
        if(present(pivotcols)) pivotcols(rowi) = colj
        ! Eliminate the rows
        !!$OMP PARALLEL DO PRIVATE(i)
        do i=1,n
            if(i/=rowi .and. Amat(i, colj)>0) then
                Amat(i, :) = ieor(Amat(i, :), Amat(rowi, :))
            end if
        end do
        !!$OMP END PARALLEL DO
        rowi = rowi + 1
        if (rowi > n) then
            exit
        end if
200         continue
    end do
    rank = rowi - 1
end subroutine

subroutine code_dual(G, H, iscols)
    integer :: n, k, nmk
    integer(1), dimension(:, :), intent(inout) :: G ! k x n
    integer(1), dimension(:, :), intent(out) :: H ! (n-k) x n
    integer(4), dimension(:), intent(out), optional :: iscols ! k
    integer rank, i, j, j2, l
    integer(1) tmp
    integer, dimension(:), allocatable :: pivtcols, stdcols
    integer(1), dimension(:), allocatable :: coltypes ! coltype(i) = 1 if i is an identity column of G in standard form

    k = size(G, 1)
    n = size(G, 2)
    nmk = size(H, 1)

    if(nmk /= n-k .or. size(H, 2) /= n) then
        print '(A)', "Invalid dimensions for H (code_dual)."
        error stop
    end if

    allocate(pivtcols(k))
    allocate(stdcols(nmk))
    allocate(coltypes(n))
    ! Perform gaussian elimination. List the pivot(identity) and standard columns of G.
    call gaussian_elim(G, rank, pivotcols=pivtcols)
#ifndef NDEBUG
    print '(A)', "G Pivot columns"
    print '(*(I4))', pivtcols(:)
#endif
    if(rank /= k) then
        print '(A, I8, I8)', "Assertion failed (rank(G) == dim(H, 1) ), ", rank, nmk
        error stop
    end if
    coltypes(:) = 0
    do i = 1, k
        coltypes(pivtcols(i)) = 1
    end do
    i=1
    do j = 1, n
        if(coltypes(j)==0) then
            stdcols(i) = j
            i = i + 1
        end if
    end do
    H(:, :) = 0
    ! assign the identity columns of H
    do j2 = 1, nmk
        j = stdcols(j2)
        do i = 1, nmk
            if (i == j2 )then
                H(i, j) = 1
            else
                H(i, j) = 0
            end if
        end do
    end do
    ! assign the standard columns of H
    do j2 = 1, k
        j = pivtcols(j2)
        do i = 1, nmk
            H(i, j) = G(j2, stdcols(i))
        end do
    end do
    ! Assert H G^T == 0
    do i = 1, n-k
        do j = 1, k
            tmp = 0
            do l = 1, n
                tmp = ieor(tmp, iand(H(i, l), G(j, l)))
            end do
            if(tmp /= 0) then
                print '(A)', "Assertion failed (H G^T == 0)."
                error stop
            end if
        end do
    end do
#ifndef NDEBUG
    print '(A,I4,A,I4,A)', "n = ",n, " k = ",k," Parity check constructed."
#endif
    if(present(iscols)) then
        iscols(:) = pivtcols(:)
    end if
end subroutine code_dual
    
subroutine isd_decode(G,  y, isd_cols, x)
    integer(1), dimension(:,:), intent(in) :: G
    integer(1), dimension(:), intent(in) ::  y
    integer, dimension(:), intent(in) :: isd_cols
    integer(1), dimension(:), intent(out) :: x

    integer(1), dimension(:, :), allocatable :: Gaug
    integer :: k, n, i, r
    k = size(G, 1)
    n = size(G, 2)
    allocate(Gaug(k, k+1))

    do i = 1, k
        Gaug(i, 1:k) = G(1:k, isd_cols(i))
    end do
    do i = 1, k
        Gaug(i, k+1) = y(isd_cols(i))
    end do

    call gaussian_elim(Gaug, r)
    if(r /= k) then
        print '(A)', "Assertion failed: r==k (isd_decode)"
        error stop
    end if

    x(:) = Gaug(:, k+1)
    
end subroutine isd_decode

end module lincode
