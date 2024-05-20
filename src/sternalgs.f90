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

module sternalgs
    use stern, only: SternAtk, SternComboWrk, L0
    implicit none
    !integer, public, parameter :: L0 = 8
contains
    subroutine make_stern_combos(st, wrk)
        ! Step 1 of the Stern attack:
        ! Initialize the SternComboWrk object and construct all p-combinations of the split information set.

        use randutil
        use mathutil
        type(SternAtk), intent(inout) :: st
        integer(4) :: l, p, m
        type(SternComboWrk), intent(inout) :: wrk
        type(ComboIter), dimension(2) :: combo_iter
        integer(8) x, n
        integer(8) i, k, khalf, cidx
        
        l = wrk%l
        p = wrk%p
        m = wrk%m
        k = st%ncode
        n = st%nvars
        khalf = k / 2
        ! update the information set and randomly split
        call st%sternatk_set_workisd()
        call ishuffle(st%workisd)
        ! #ifndef NDEBUG
        !         print '(A,I4,A,I8,I8)', "Constructing combinations: p=", p, " K=",wrk%kx(1),wrk%kx(2)
        !         print '(A, I8, I8)', "Number of combinations: ", wrk%nx(1), wrk%nx(2)
        !         print '(A)', "Split 1:"
        !         print '(*(I4))', st%workisd(1:khalf)
        !         print '(A)', "Split 2:"
        !         print '(*(I4))', st%workisd(khalf+1:k)
        ! #endif
        call combo_iter(1)%comboiter_init(wrk%kx(1), p)
        call combo_iter(2)%comboiter_init(wrk%kx(2), p)

        do x=1,2
            cidx = 1
            ! This do loop level could be parallelized over omp.
            ! It will require manually chunking cidx and initializing combo_iter seperately with lin_to_combo
            do
                wrk%combos(:, cidx, x) = st%G(:, st%workisd(1 + (x-1)*khalf + combo_iter(x)%c(1))  )
                ! todo: use the sequentiality of the combo iterator to cache additions
                do i=2,p
                    wrk%combos(:, cidx, x) = ieor(wrk%combos(:, cidx, x), &
                            st%G(:, st%workisd(1 + (x-1)*khalf + combo_iter(x)%c(i))  ) )
                end do
                if(.not. combo_iter(x)%comboiter_next()) exit
                cidx = cidx + 1
            end do
        end do
    end subroutine 

    subroutine collision_search(st, wrk, w, mfound)
        ! Step 2
        ! Look for collisions on small subsets of the redundancy set
        use randutil
        use mathutil
        type(SternAtk), intent(inout) :: st
        type(SternComboWrk), intent(inout) :: wrk
        integer, intent(in) :: w
        integer, intent(out) :: mfound
        type(ComboIter), dimension(2) :: combo_iter
        integer p, l, n, m, m2
        integer i, i2, j, k, khalf
        integer hw
        integer(1) tmp1
        p = wrk%p
        l = wrk%l
        m = wrk%m
        k = st%ncode
        n = st%nvars
        khalf = k / 2
        wrk%isdsols(:, :) = 0
        mfound = 0

        do m2 =1,m
            call rand_choice(st%nclauses, l, wrk%colpos(:, m2))
        end do

        do m2=1,m
            do i=1,wrk%nx(1)
                do j=1,wrk%nx(2)
                    hw = 0
                    do i2=1,l
                        tmp1 = ieor(wrk%combos(wrk%colpos(i2, m2), i, 1), &
                                wrk%combos(wrk%colpos(i2, m2), j, 2))
                        if(st%naux>0) tmp1 = ieor(tmp1, st%G(wrk%colpos(i2, m2), n+1))
                        if(tmp1 /= 0) hw = hw + 1
                    end do
                    if(hw == 0) then ! Collision detected
                        wrk%colwork(:, m2) = ieor( &
                            wrk%combos(:, i, 1), &
                            wrk%combos(:, j, 2) )
                        ! Get hamming distance from y if it is provided
                        if(st%naux>0) wrk%colwork(:, m2) = ieor( &
                                wrk%colwork(:, m2), &
                                st%G(:,n+1))
                        hw = 0 ! get the total Hamming weight
                        do i2=1,st%nclauses
                            hw = hw + wrk%colwork(i2, m2)
                        end do
                        ! We found the solution!
                        if(hw <= w-2*p) then
                            ! w-2p errors in the redundancy set
                            wrk%bestsols(:, m2) = 0
                            do i2=1,st%nclauses
                                if(wrk%colwork(i2, m2)>0) wrk%bestsols(st%rdncols(i2), m2) = 1
                            end do
                            ! Get the combinatorial index from the linear indices
                            call lin_to_combo(i-1, p, combo_iter(1)%c)
                            call lin_to_combo(j-1, p, combo_iter(2)%c)
                            do i2=1,p
                                wrk%isdsols(i2, m2) = st%workisd(1 + combo_iter(1)%c(i2))
                                wrk%isdsols(p+i2, m2) = st%workisd(1 + khalf + combo_iter(2)%c(i2))
                            end do

                            print '(A)'," ** Significant Collision **"
                            print '(A2,I6,A,*(I2))',"C1", i," | ",  wrk%combos(:, i, 1)
                            print '(A2,I6,A,*(I2))',"C2", j," | ",  wrk%combos(:, j, 2)

                            print '(A4,*(I4))', "C1: ", wrk%isdsols(1:p, m2)
                            do i2=1,p
                                wrk%bestsols(wrk%isdsols(i2, m2), m2) = 1
                                print '(I8,A,*(I2))', wrk%isdsols(i2, m2), " | ", &
                                        st%G(:, wrk%isdsols(i2, m2) )
                            end do

                            print '(A4,*(I4))', "C2: ", wrk%isdsols(p+1:2*p, m2)
                            do i2=1,p
                                wrk%bestsols(wrk%isdsols(p+i2, m2), m2) = 1
                                print '(I8,A,*(I2))', wrk%isdsols(p+i2, m2), " | ", &
                                        st%G(:, wrk%isdsols(p+i2, m2) )
                            end do
                            print '(A8,A,*(I2))', "dy"," | ", wrk%colwork(:, m2)
                            print '(A)', " ** Solution ** "
                            print '(*(I2))', wrk%bestsols(:, m2)
                            mfound = m2
                            return
                        end if
                    end if
                end do
            end do
            if(l<=0) exit;
        end do
    end subroutine

    subroutine collision_search_p1(st, wrk, w, mfound)
        ! Look for collisions on small subsets of the redundancy set
        ! Specialize combined steps 1 & 2 for p=1 
        use randutil
        use mathutil
        !use stern, only : L0
        type(SternAtk), intent(inout) :: st
        type(SternComboWrk), intent(inout) :: wrk
        integer, intent(in) :: w
        integer, intent(out) :: mfound
        integer p, l, n, m, m2
        integer i, i2, i3, j, k, khalf, x, a, b
        integer hw
        integer(1) tmp1
        integer(1), dimension(L0) :: atmp
        p=1
        l = wrk%l
        m = wrk%m
        k = st%ncode
        n = st%nvars
        khalf = k / 2
        wrk%isdsols(:, :) = 0
        mfound = 0
        !print '(A,I4,A,I4,A,I4)', "p=1, w=", w, "ncl=",st%nclauses," naux=",st%naux
        ! update the information set and randomly partition it
        call st%sternatk_set_workisd()
        call ishuffle(st%workisd)
        !print '(*(I3))', st%workisd
        ! copy the shuffled columns into the work array
        do x=1,2
            do i = 1, wrk%kx(x)
                wrk%combos(:, i, x) = st%G(:, st%workisd((x-1)*khalf + i)  )
            end do
        end do
        !print '(A,*(I3))', "Rdn: ", st%rdncols(:)
        !print('(A,*(I3))'), " I1: ", st%workisd(1:khalf)
        !print('(A,*(I3))'), " I2: ", st%workisd(khalf+1:k)
        ! randomly select l redundancy bits
        if(l>0 .and. .not. wrk%qcol) then
            do m2 =1,m
                call rand_choice(st%nclauses, l, wrk%colpos(:, m2))
            end do
            !print('(A,*(I3))'), "lbits: ", wrk%colpos(:, m2)
        end if
        do m2 = 1, m
            do i = 1, wrk%kx(1)
                do j = 1, wrk%kx(2)
                    if(l>0) then ! Original stern
                        hw = 0
                        if(wrk%qcol) then
                            do i2 = 1, l
                                a=1+(i2-1)*L0 + (m2-1)*l*L0
                                b=i2*L0 + (m2-1)*l*L0
                                atmp = ieor( wrk%combos(a:b, i, 1), wrk%combos(a:b, j, 2) )
                                if(st%naux>0) atmp = ieor(atmp, st%G(a:b, n+1))
                                do i3=1,L0
                                    hw = hw + atmp(i3)
                                end do
                            end do
                        else
                            do i2=1,l
                                tmp1 = ieor(wrk%combos(wrk%colpos(i2, m2), i, 1), &
                                            wrk%combos(wrk%colpos(i2, m2), j, 2))
                                if(st%naux>0) tmp1 = ieor(tmp1, st%G(wrk%colpos(i2, m2), n+1))
                                !if(tmp1 /= 0) goto 200 ! continue loop (non-zero HW) ! this increases CPU time somewhat 
                                if(tmp1 /= 0) hw = hw + 1
                            end do
                        end if
                        if(hw>0) goto 200
                    end if
                    ! Hamming weight was zero on l redundancy bits
                    ! Now calculate the full Hamming weight
                    wrk%colwork(:, m2) = ieor( wrk%combos(:, i, 1), wrk%combos(:, j, 2) )
                    ! Get hamming distance from y if it is provided
                    if(st%naux>0) &
                        wrk%colwork(:, m2) = ieor( wrk%colwork(:, m2), st%G(:,n+1))
                    hw = 0 ! get the total Hamming weight
                    do i2=1,st%nclauses
                        hw = hw + wrk%colwork(i2, m2)
                    end do
                    !write(*, '(I4)', advance='no') hw
                    if (hw > w-2*p) goto 200 ! continue loop (HW too large)
                    !write(*, '(A)') '!'
                    ! At this point, we found the solution with the desired Hamming weight
                    ! w-2p errors in the redundancy set
                    wrk%bestsols(:, m2) = 0
                    do i2=1,st%nclauses
                        if(wrk%colwork(i2, m2)>0) wrk%bestsols(st%rdncols(i2), m2) = 1
                    end do
                    wrk%isdsols(1, m2) = st%workisd(i)
                    wrk%isdsols(2, m2) = st%workisd(khalf + j)
                    wrk%bestsols(wrk%isdsols(1, m2), m2) = 1
                    wrk%bestsols(wrk%isdsols(2, m2), m2) = 1
                    mfound = m2
                    return
200             continue
                end do
            end do
        end do
        ! write(*,*)
        
    end subroutine collision_search_p1
end module sternalgs

