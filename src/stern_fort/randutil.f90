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

module randutil
  implicit none
contains
  function lcg(s)
    use iso_fortran_env, only : int64
    ! This simple PRNG might not be good enough for real work, but is
    ! sufficient for seeding a better PRNG.
    integer :: lcg
    integer(int64), intent(inout) :: s
    if (s == 0) then
      s = 104729
    else
      s = mod(s, 4294967296_int64)
    end if
    s = mod(s * 279470273_int64, 4294967291_int64)
    lcg = int(mod(s, int(huge(0), int64)), kind(0))
  end function lcg

  subroutine init_random_seed(iseed, addentr)
    ! based on https://gcc.gnu.org/onlinedocs/gcc-4.9.1/gfortran/RANDOM_005fSEED.html
    ! Initialize the global RNG seed
    use iso_fortran_env, only : int64
#ifdef IFORT
    use ifport
#endif
    integer, allocatable :: aseed(:)
    integer :: i, n, un, istat, dt(8), pid
    integer(int64), intent(in), optional :: iseed
    integer(int64), intent(in), optional :: addentr
    integer(int64) t

    call random_seed(size = n)
    allocate(aseed(n))
    if(.not. present(iseed)) then
      ! First try if the OS provides a random number generator
      open(newunit = un, file = "/dev/urandom", access = "stream", &
          form = "unformatted", action = "read", status = "old", iostat = istat)
      if (istat == 0) then
        read(un) aseed
        close(un)
      else
        ! Fallback to XOR:ing the current time and pid. The PID is
        ! useful in case one launches multiple instances of the same
        ! program in parallel.
        call system_clock(t)
        if (t == 0) then
          call date_and_time(values = dt)
          t = (dt(1) - 1970) * 365_int64 * 24 * 60 * 60 * 1000 &
              + dt(2) * 31_int64 * 24 * 60 * 60 * 1000 &
              + dt(3) * 24_int64 * 60 * 60 * 1000 &
              + dt(5) * 60 * 60 * 1000 &
              + dt(6) * 60 * 1000 + dt(7) * 1000 &
              + dt(8)
        end if
        pid = getpid()
        t = ieor(t, int(pid, kind(t)))
        if(present(addentr)) t = ieor(t, addentr)
        do i = 1, n
          aseed(i) = lcg(t)
        end do
      end if
    else
      t = iseed
      if(present(addentr)) t = ieor(t, addentr)
      do i = 1, n
        aseed(i) = lcg(t)
      end do
    end if
    call random_seed(put = aseed)
  end subroutine init_random_seed

  subroutine rand_int(b, x, a)
    ! Return a random integer in the range [a, b-1]
    integer(8), intent(in) :: b
    integer(8), intent(out) :: x
    integer(8), intent(in), optional :: a
    integer(8) a_
    real r
    if(present(a)) then
      a_ = a
    else
      a_ = 0
    end if
    call random_number(r)
    x = a_ + floor(real(b - a_) * r)
  end subroutine

  subroutine rand_choice(n, k, out)
    ! Randomly choose k indices from the range 1:n without replacement.
    ! If k is large, it might be more efficient to perform a random permutation
    integer, intent(in) :: n, k
    integer :: i, j
    integer(8) ir
    integer, intent(out), dimension(k) :: out
    do i = 1, k
      out(i) = 0
    end do
    i = 1
    do
      call rand_int(int(n + 1, 8), ir, a = 1_8)
      do j = 1, i - 1
        if(out(j) == ir) goto 100
      end do
      out(i) = int(ir, 4)
      i = i + 1
      if(i>k) exit
      100     continue
    end do
  end subroutine rand_choice

  subroutine ishuffle(arr)
    integer, intent(inout), dimension(:) :: arr
    integer :: n, i, tmp
    integer(8) :: j

    n = size(arr, 1)
    do i = 1, n - 1
      call rand_int(int(n + 1, 8), j, int(i, 8))
      if(j>i) then
        tmp = arr(j)
        arr(j) = arr(i)
        arr(i) = tmp
      end if
    end do
  end subroutine ishuffle

  subroutine jshuffle(arr)
    integer(8), intent(inout), dimension(:) :: arr
    integer(8) :: n, i, j, tmp

    n = size(arr, 1)
    do i = 1, n - 1
      call rand_int(n + 1, j, i)
      if(j>i) then
        tmp = arr(j)
        arr(j) = arr(i)
        arr(i) = tmp
      end if
    end do
  end subroutine jshuffle

end module randutil
