!Author: Humberto Munoz Bauza (humberto.munozbauza@nasa.gov)
!
!Copyright © 2023, United States Government, as represented by the Administrator
!of the National Aeronautics and Space Administration. All rights reserved.
!
!The PySA, a powerful tool for solving optimization problems is licensed under
!the Apache License, Version 2.0 (the "License"); you may not use this file
!except in compliance with the License. You may obtain a copy of the License at
!http://www.apache.org/licenses/LICENSE-2.0.
!
!Unless required by applicable law or agreed to in writing, software distributed
!under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
!CONDITIONS OF ANY KIND, either express or implied. See the License for the
!specific language governing permissions and limitations under the License.

module sterncmod
    use iso_c_binding
    implicit none

    type, bind(C), public :: SterncOpts
        integer(c_int8_t) :: parcheck 
        integer(c_int8_t) :: bench
        integer(c_int8_t) :: test_hw1
        ! true if a parity check matrix ((n-k) x n)is passed. 
        ! otherwise, it is the generator matrix transpose ( n x k )
        integer(c_int32_t) :: nvars ! code size
        integer(c_int32_t) :: nclauses
        integer(c_int32_t) :: t ! error weight
        integer(c_int32_t) :: max_iters
        integer(c_int32_t) :: l
        integer(c_int32_t) :: p
        integer(c_int32_t) :: m
        integer(c_int32_t) :: block_size
    end type SterncOpts

contains
    subroutine cptr_to_arr(cparr, farr)
        ! Copy the contents of a C array containing a row-major matrix into a 2D fortran array
        integer(1), intent(in) :: cparr(*)
        integer(1), dimension(:, :), intent(out) :: farr
        integer :: a, b, i, j
        a = size(farr, 1)
        b = size(farr, 2)
        do i = 1,a
            do j =1,b
                farr(i, j) = cparr(1 + (i-1)*b + (j-1))
            end do
        end do
    end subroutine

    subroutine into_parcheck(codearr,  opts, newarr, iscols)
        use lincode
        ! Convert a generator matrix problem into a parity check problem, if not already one
        integer(1), dimension(:, :), intent(in) :: codearr ! n x (k+naux)
        integer(1), dimension(:, :), intent(out) :: newarr ! (n-k)x(n+naux)
        integer(1), dimension(:, :), allocatable :: tmparr 
        integer, dimension(:), intent(out), optional :: iscols
        integer, dimension(:), allocatable :: iscols_
        integer(c_int32_t) :: n, k, nmk, naux, i, j
        type(SternCOpts), intent(inout) ::  opts
        if(opts%parcheck>0) then ! nothing to do
            newarr(:, :) = codearr(:, :)
        end if

        n = opts%nclauses
        k = opts%nvars
        nmk = n-k
        naux = size(codearr, 2) - k 
        allocate(tmparr(k, n))
        do i =1,k
            do j = 1, n
                tmparr(i,j) = codearr(j, i)
            end do
        end do

        call code_dual(tmparr, newarr(1:nmk,1:n), iscols_)
        if(present(iscols)) iscols(:) = iscols_(:)
        ! Apply the parity check matrix on the auxiliary vectors
        do i = 1, naux
            do j = 1,nmk
                newarr(j,n+i) = iand(sum(newarr(j,1:n)*codearr(:,k+i)), 1_1)
            end do
        end do
        ! Update the variables/clauses counts
        opts%nvars = n
        opts%nclauses = nmk

    end subroutine into_parcheck
end module sterncmod

subroutine sternc(cinarr, sternopts)
    use iso_c_binding
    use iso_fortran_env
    use lincode
    use sterncmod
    use stern
    use sternalgs
#ifdef USEMPI
    use mpi_f08 
#endif
    use randutil 
    use mathutil
    use iso_fortran_env, only : output_unit
    implicit none
    integer(kind=c_int8_t), intent(in) :: cinarr(*)
    type(SterncOpts), intent(in) :: sternopts ! original input options
    type(SterncOpts) opts ! input options converted to parity check form, if necessary
    integer(kind=c_int32_t):: numvars, numclauses, t
    integer(kind=c_int32_t) :: max_iters, l_, p, m
    integer k
    integer l
    !integer(8) memsize
    real :: tstart, tend, dt, dt_buf ! CPU time start and end, rank CPU time, and total MPI CPU time
    real(8) :: psucc, tperit, tts ! success probability, time per iteration, time to solution
    type(SternAtk) :: stern_atk
    type(SternComboWrk) :: wrk
    !type(SternAtk), dimension(:), allocatable :: stern_array
    integer           :: rank,  master_rank
    integer :: i, j, mfound, nsuccess, niters, niters_buf
    ! Completion data (rank, iteration, m_index)
    ! Synchronized over MPI ranks if not bench marking
    integer(8), dimension(3) :: compl
    integer(1), dimension(:,:), allocatable :: inputarr, garr, harr
    integer(1), dimension(:), allocatable :: stern_solution, code_solution
    integer, dimension(:), allocatable :: isd_sol
#ifdef USEMPI
    integer ierr
    logical mpiflag
    integer :: rc, nproc
    integer :: nsucc_buf
    integer(8), dimension(3) :: compl_buf
    integer, dimension(:), allocatable :: rank_nsuccesses
    type(MPI_Request) :: mpireq, mpireq2
#endif
    ! call mpi_init(rc)
    master_rank = 0
    ! Get number of active processes.
#ifdef USEMPI
    call mpi_comm_size(mpi_comm_world, nproc, rc)
    ! Identify process.
    call mpi_comm_rank(mpi_comm_world, rank, rc)
#else
        rank = 0
#endif
    ! MPI may not initialize different seeds for each rank
    call init_random_seed(addentr=int(rank,8))

    opts = sternopts
    ! Copy the data into a fortran array
    allocate(inputarr(opts%nclauses, opts%nvars+1))
    call cptr_to_arr(cinarr, inputarr)
#ifndef NDEBUG
    if(opts%parcheck>0) then
        print '(A)', "Parity Check Matrix"
    else
        print '(A)', "Generator Matrix"
    end if
    do i=1,opts%nclauses
        print '(*(I2))', inputarr(i, :)
    end do
#endif
    ! Convert the problem to parity check form if necessary
    if(opts%parcheck>0) then ! H input. n = numvars, n-k = numclauses, 1 aux column
        allocate(harr(opts%nclauses, opts%nvars+1))
        harr(:, :) = inputarr(:, :)
    else ! GT input. Here, n = nclauses, k = nvars, 1 aux column
        allocate(harr(opts%nclauses - opts%nvars, opts%nclauses+1))
        allocate(garr(opts%nvars, opts%nclauses)) ! [k, n]
        do i = 1, opts%nvars
            do j = 1, opts%nclauses
                garr(i, j) = inputarr(j, i)
            end do 
        end do
       
        call into_parcheck(inputarr, opts, harr)
#ifndef NDEBUG
        print '(A)', "Converting into parity check form"
        do i=1,opts%nclauses
            print '(*(I2))', harr(i, :)
        end do
#endif
    end if

    numvars = opts%nvars
    numclauses = opts%nclauses
    t = opts%t
    max_iters = opts%max_iters
    l_ = opts%l
    p = opts%p
    m = opts%m
    k = numvars - numclauses

    allocate(stern_solution(numvars))
    allocate(isd_sol(k))
    allocate(code_solution(k))
    !print '(A)', "Calling sternc"
    if(l_>=L0) then
        if(mod(l_, L0) /= 0) then
            if(rank==0) print '(A, I8)', "Adjusted l to ", (l_/L0)*L0
            l=(l_/L0)*L0
        else
            if(rank==0) print '(A, I8)', "Using l=", l_
            l=l_
        end if
    elseif(l_>0) then
        if(rank==0) print '(A, I8)', "Adjusted l to ", L0
        l=L0
    else 
        l=l_
    end if
    flush(output_unit)
    
    ! Initialize working arrays for Stern algorithm
    call stern_from_harr(stern_atk, harr, naux=1)
    
    call wrk%sterncombowrk_constructor(numvars, stern_atk%ncode, l, p, m, .true.)
#ifdef USEMPI
    if (rank == master_rank) then
        call MPI_Irecv(compl_buf, 3, MPI_INTEGER8, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, mpireq, ierr)
    else
        call MPI_Ibcast(compl_buf, 3, MPI_INTEGER8, master_rank, MPI_COMM_WORLD, mpireq, ierr)
    end if
#endif
    compl(1) = 0
    compl(2) = -1
    compl(3) = -1
    nsuccess = 0
    niters = 0
    call cpu_time(tstart)
    do i=1,max_iters
        call stern_atk%sternatk_restart_rdn()
        if(p==1) then
            call collision_search_p1(stern_atk, wrk, t, mfound)
        else
            call make_stern_combos(stern_atk, wrk)
            call collision_search(stern_atk, wrk, t, mfound)
        end if
        ! call stern_atk%sternatk_combo_iter(wrk, t, mfound)
#ifdef USEMPI
        ! If master rank, check if another rank has finished
        if(opts%bench==0) then
            if(rank == master_rank) then
                call MPI_Test(mpireq, mpiflag, MPI_STATUS_IGNORE, ierr)
                if(mpiflag) then
                    !write(*, '(A,I8,A,2(I8))') "MPI", rank, ": received ", compl_buf
                    if(compl_buf(3) > 0) then ! broadcast successful completion from another rank and terminate
                        compl(:) = compl_buf(:)
                        call MPI_Ibcast(compl, 3, MPI_INTEGER8, master_rank, MPI_COMM_WORLD, mpireq2, ierr)
                        call MPI_Wait(mpireq2, MPI_STATUS_IGNORE, ierr)
                        !write(*, '(A,I8,A,2(I8))') "MPI", rank, ": broadcasted ", compl
                        exit
                    else
                        call MPI_Irecv(compl_buf, 3, MPI_INTEGER8, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, mpireq, ierr)
                    end if
                end if
            else ! Check if the master rank has broadcasted completion
                call MPI_Test(mpireq, mpiflag, MPI_STATUS_IGNORE, ierr)
                if(mpiflag) then
                    !write(*, '(A,I8,A,2(I8))') "MPI ", rank, ": received broadcast ", compl_buf
                    compl(:) = compl_buf(:)
                    exit
                end if
            end if
        end if
#endif
        ! There can be a deadlock condition if the master rank and a worker rank both simultanously find the solution,
        ! check MPI_Test, and then worker hangs on MPI_Send while master hangs on MPI_Wait. This is extremely unlikely
        ! but might be worth addressing.
        if(mfound > 0) then
            ! write(*, '(A,I8,A,I8,A)') "Rank ", rank, " found solution after ", i, " iterations."
            ! print '(*(I2))', wrk%bestsols(:, mfound)
            stern_solution(:) = wrk%bestsols(:, mfound)
            isd_sol(:) = stern_atk%workisd(:)
            compl(1) = rank
            compl(2) = i
            compl(3) = mfound
            !exit ! Benchmark if not using MPI
            nsuccess = nsuccess + 1
            niters = i
#ifdef USEMPI
            if(rank == master_rank .and. opts%bench==0) then
                !write(*, '(A,I8,A,2(I8))') "MPI ", rank, ": broadcasting ", compl
                call MPI_Ibcast(compl, 3, MPI_INTEGER8, master_rank, MPI_COMM_WORLD, mpireq2, ierr)
                call MPI_Wait(mpireq2, MPI_STATUS_IGNORE, ierr)
                !write(*, '(A,I8,A)') "MPI ", rank, ": broadcast completed."
            else ! If not master rank, send the termination signal to the master rank
                !write(*, '(A,I8,A,2(I8))') "MPI ", rank, ": sending ", compl
                call MPI_Send(compl, 3, MPI_INTEGER8, master_rank, 1, MPI_COMM_WORLD, ierr)
                !write(*, '(A,I8,A)') "MPI ", rank, ": send completed."
            end if
#endif 
            if(opts%bench == 0) exit 

        end if
        niters = i
    end do
    call cpu_time(tend)
    dt = tend - tstart
#ifdef USEMPI
    ! Wait for all ranks to finish
    if(opts%bench>0) then
        call MPI_Allreduce(nsuccess, nsucc_buf, 1, MPI_INTEGER8, MPI_SUM, MPI_COMM_WORLD, ierr)
        !nsuccess = nsucc_buf
        niters = max_iters * nproc
        if(nsucc_buf>0) then ! transmit the solution to the master rank
            allocate(rank_nsuccesses(nproc))
            call MPI_Allgather(nsuccess, 1, MPI_INTEGER, rank_nsuccesses, 1, MPI_INTEGER, MPI_COMM_WORLD, ierr)
            do i = 0, nproc-1
                if(rank_nsuccesses(i+1)>0) then
                    if(i/=master_rank) then
                        if(rank==master_rank)  &
                            call MPI_Recv(stern_solution, numvars, MPI_INTEGER1, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
                        if(rank==i)  &
                            call MPI_Send(stern_solution, numvars, MPI_INTEGER1, master_rank, 2, MPI_COMM_WORLD, ierr)
                        if(rank==master_rank) &
                            call MPI_Recv(compl, 3, MPI_INTEGER8, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE, ierr)
                        if(rank==i) &
                            call MPI_Send(compl, 3, MPI_INTEGER8, master_rank, 3, MPI_COMM_WORLD, ierr)
                    end if
                    exit
                end if
            end do
        end if
        nsuccess = nsucc_buf
    else
        call MPI_Barrier(MPI_COMM_WORLD, ierr)
        ! synchronize one more time to break if two ranks found a solution simultaneously before
        ! the master rank could broadcast
        call MPI_Bcast(compl, 3, MPI_INTEGER8, master_rank, MPI_COMM_WORLD, ierr) 
        call MPI_Bcast(stern_solution, numvars, MPI_INTEGER1, int(compl(1), 4), MPI_COMM_WORLD, ierr)
        call MPI_Bcast(isd_sol, k, MPI_INTEGER, int(compl(1), 4), MPI_COMM_WORLD, ierr )
        call MPI_Reduce(niters, niters_buf, 1, MPI_INTEGER8, MPI_SUM, master_rank, MPI_COMM_WORLD, ierr)
        if(rank==master_rank .and. compl(3)>0)  &
            print '(A,I10,A,I10)', "Solution found in Rank ", compl(1), " Iter ", compl(2)
    end if
    call MPI_Reduce(dt, dt_buf, 1, MPI_REAL4, MPI_SUM, master_rank, MPI_COMM_WORLD, ierr)
#else
    dt_buf = dt 
    niters_buf = niters
#endif
   
    if(rank==master_rank) then
        print '(A, F20.6)', "CPU Time (s): ", dt_buf
        if(opts%bench>0) then
            tperit = real(dt_buf/real(niters, 8), 8)
            print '(A, F10.3)', "t/iter (us): ", tperit*1000000
            print '(A, I10, A, I10)', "Success Count: ", nsuccess, " / ", niters
            psucc = real(nsuccess, 8) / real(niters, 8)
            tts = tperit * (log(1.0_8 -0.99_8)/log(1.0_8 - psucc))
            print '(A, E14.6)', "Success Prob: ", psucc
            print '(A, F14.6)', "TTS(99%) (s): ", tts
            print '(A, F14.6)', "lg TTS(99%) : ", log2(tts)
        else
            print '(A, I10)', "Iterations: ", niters_buf
        end if
    end if

    if(rank==master_rank ) then
        if(compl(3) > 0) then
            print '(A)', "Error vector ="
            call print_f2_array(stern_solution)
            print '(*(I2))', stern_solution
            
            if(sternopts%parcheck==0) then
                stern_solution = ieor(stern_solution, inputarr(:, k+1))
                !print '(*(I2))', stern_solution
                !call print_f2_array(stern_solution)
                !print '(*(I4))', isd_sol
                call isd_decode(garr, stern_solution, isd_sol, code_solution)
                print '(A)', "Decoded vector ="
                call print_f2_array(code_solution)
                print '(*(I2))', code_solution
            end if
        else
            print '(A)', "No solution"
        end if
    end if
    
    

    !call mpi_finalize(rc)

end subroutine sternc
