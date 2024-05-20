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

program tests
    use randutil
    use mathutil
    implicit none

    real r
    integer(4), dimension(4) :: arr
    integer(4), dimension(:), allocatable :: c2
    integer(4) n, i, k
    logical iter
    type(ComboIter) combo_iter
    n = 7
    k = 4
    allocate(c2(k))
    call random_number(r)
    print '(A, 10F5.4)', "Uninitialized seed: ", r

    call init_random_seed(1234_8)
    call random_number(r)
    print '(A, 10F5.4)', "Seed=1234: ", r
    do i = 1,8
        call rand_choice(n, k, arr)
        write(*,'(*(I2))') arr
        call shuffle(arr)
        write(*,'(*(I2))') arr
    end do

    print '(A)', "Combo iter:"

!    do
!        write(*,'(6X,*(I2))') combo_iter%c(1:k)
!        iter=combo_iter%comboiter_next()
!        if(.not. iter) exit
!    end do

    call combo_iter%comboiter_init(n,k)
    do i=0,n_C_r(int(n,8), int(k,8))
        call lin_to_combo(i, k, c2(1:k))
        write(*,'(I6,*(I2))') i, c2(1:k)
        write(*,'(6X,*(I2))') combo_iter%c(1:k)
        iter=combo_iter%comboiter_next()
    end do

    call init_random_seed()
    call random_number(r)
    print '(A, 10F5.4)', "Reseed from entropy: ", r

end program tests