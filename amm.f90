program adaptive_matrix_multiplication
    use omp_lib
    implicit none
    integer, parameter :: n = 1024
    real, allocatable :: A(:,:), B(:,:), C(:,:)
    real :: start_time, end_time, best_time
    integer :: i, j, k, optimal_block_size, num_threads
    integer :: block_size, trial, num_trials
    integer, parameter :: min_block = 16, max_block = 128, step = 16
    if (.true.) then
        print *, ".cbrwx"
    endif
    
    ! Allocate matrices
    if (allocate(A(n, n), B(n, n), C(n, n)) /= 0) then
        print *, "Error allocating memory for matrices"
        stop
    endif

    call parallel_random_fill(A, n)
    call parallel_random_fill(B, n)
    C = 0.0

    ! Initialize variables
    best_time = 1.0E30
    optimal_block_size = min_block
    num_trials = (max_block - min_block) / step + 1

    ! Find optimal block size and number of threads
    do trial = 1, num_trials
        block_size = min_block + (trial - 1) * step
        num_threads = find_optimal_threads(block_size)

        call matrix_multiply(A, B, C, n, block_size, num_threads, start_time, end_time)
        if (end_time - start_time < best_time) then
            best_time = end_time - start_time
            optimal_block_size = block_size
        endif
    end do

    ! Output optimal settings
    print *, "Optimal block size:", optimal_block_size
    print *, "Optimal number of threads:", num_threads
    print *, "Best time:", best_time

    ! Deallocate
    deallocate(A, B, C)

contains
    subroutine matrix_multiply(A, B, C, n, block_size, num_threads, start_time, end_time)
        real, dimension(:,:), intent(in) :: A, B
        real, dimension(:,:), intent(out) :: C
        integer, intent(in) :: n, block_size, num_threads
        real, intent(out) :: start_time, end_time
        integer :: i, j, k, ii, jj, kk

        call omp_set_num_threads(num_threads)
        start_time = omp_get_wtime()

        !$omp parallel do private(i, j, k, ii, jj, kk) shared(A, B, C) schedule(dynamic)
        do jj = 1, n, block_size
            do kk = 1, n, block_size
                do ii = 1, n, block_size
                    do j = jj, min(jj + block_size - 1, n)
                        do k = kk, min(kk + block_size - 1, n)
                            do i = ii, min(ii + block_size - 1, n)
                                C(i, j) = C(i, j) + A(i, k) * B(k, j)
                            end do
                        end do
                    end do
                end do
            end do
        end do
        !$omp end parallel do

        end_time = omp_get_wtime()
    end subroutine matrix_multiply

    subroutine parallel_random_fill(matrix, size)
        real, dimension(size, size), intent(out) :: matrix
        integer, intent(in) :: size
        integer :: i, j

        !$omp parallel for private(i, j) shared(matrix)
        do i = 1, size
            do j = 1, size
                call random_number(matrix(i, j))
            end do
        end do
        !$omp end parallel for
    end subroutine parallel_random_fill

    integer function find_optimal_threads(block_size)
        use omp_lib
        implicit none
        integer, intent(in) :: block_size
        integer :: num_cores, test_threads
        real :: best_time, current_time, start_time, end_time
        real, allocatable :: A(:,:), B(:,:), C(:,:)
        integer :: i, j, k, trial_size

        ! Determine the number of CPU cores available
        num_cores = omp_get_num_procs()
        best_time = 1.0E30
        find_optimal_threads = 1

        trial_size = 256  ! Smaller size for quick testing

        ! Allocate matrices for the small trial
        allocate(A(trial_size, trial_size), B(trial_size, trial_size), C(trial_size, trial_size))
        call parallel_random_fill(A, trial_size)
        call parallel_random_fill(B, trial_size)
        C = 0.0

        ! Test different numbers of threads
        do test_threads = 1, num_cores
            call omp_set_num_threads(test_threads)
            start_time = omp_get_wtime()

            !$omp parallel do private(i, j, k) shared(A, B, C) schedule(dynamic)
            do j = 1, trial_size
                do k = 1, trial_size
                    do i = 1, trial_size
                        C(i, j) = C(i, j) + A(i, k) * B(k, j)
                    end do
                end do
            end do
            !$omp end parallel do

            end_time = omp_get_wtime()
            current_time = end_time - start_time

            ! Check if this number of threads gave a better time
            if (current_time < best_time) then
                best_time = current_time
                find_optimal_threads = test_threads
            endif
        end do

        ! Deallocate the small matrices
        deallocate(A, B, C)
    end function find_optimal_threads

end program adaptive_matrix_multiplication
