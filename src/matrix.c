#include "matrix.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Include SSE intrinsics
#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
#include <immintrin.h>
#include <x86intrin.h>
#endif

/* Below are some intel intrinsics that might be useful
 * void _mm256_storeu_pd (double * mem_addr, __m256d a)
 * __m256d _mm256_set1_pd (double a)
 * __m256d _mm256_set_pd (double e3, double e2, double e1, double e0)
 * __m256d _mm256_loadu_pd (double const * mem_addr)
 * __m256d _mm256_add_pd (__m256d a, __m256d b)
 * __m256d _mm256_sub_pd (__m256d a, __m256d b)
 * __m256d _mm256_fmadd_pd (__m256d a, __m256d b, __m256d c)
 * __m256d _mm256_mul_pd (__m256d a, __m256d b)
 * __m256d _mm256_cmp_pd (__m256d a, __m256d b, const int imm8)
 * __m256d _mm256_and_pd (__m256d a, __m256d b)
 * __m256d _mm256_max_pd (__m256d a, __m256d b)
*/

/* Generates a random double between low and high */
double rand_double(double low, double high) {
    double range = (high - low);
    double div = RAND_MAX / range;
    return low + (rand() / div);
}

/* Generates a random matrix */
void rand_matrix(matrix *result, unsigned int seed, double low, double high) {
    srand(seed);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            set(result, i, j, rand_double(low, high));
        }
    }
}

/*
 * Returns the double value of the matrix at the given row and column.
 * You may assume `row` and `col` are valid. Note that the matrix is in row-major order.
 */
double get(matrix *mat, int row, int col) {
    // Task 1.1 TODO
    return mat->data[row * mat->cols + col];
}

/*
 * Sets the value at the given row and column to val. You may assume `row` and
 * `col` are valid. Note that the matrix is in row-major order.
 */
void set(matrix *mat, int row, int col, double val) {
    // Task 1.1 TODO
    mat->data[row * mat->cols + col] = val;
}

/*
 * Allocates space for a matrix struct pointed to by the double pointer mat with
 * `rows` rows and `cols` columns. You should also allocate memory for the data array
 * and initialize all entries to be zeros. `parent` should be set to NULL to indicate that
 * this matrix is not a slice. You should also set `ref_cnt` to 1.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 */
int allocate_matrix(matrix **mat, int rows, int cols) {
    // Task 1.2 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Allocate space for the matrix data, initializing all entries to be 0. Return -2 if allocating memory failed.
    // 4. Set the number of rows and columns in the matrix struct according to the arguments provided.
    // 5. Set the `parent` field to NULL, since this matrix was not created from a slice.
    // 6. Set the `ref_cnt` field to 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows < 1 || cols < 1) {
        return -1;
    }

    struct matrix *curr = malloc(sizeof(matrix));

    if (curr == NULL) {
        return -2;
    }

    curr->data = calloc( (rows * cols), sizeof(double));

    if (curr->data == NULL) {
        return -2;
    }

    curr->rows = rows;
    curr->cols = cols;
    curr->parent = NULL;
    curr->ref_cnt = 1;

    *mat = curr;



    return 0;
}

/*
 * You need to make sure that you only free `mat->data` if `mat` is not a slice and has no existing slices,
 * or that you free `mat->parent->data` if `mat` is the last existing slice of its parent matrix and its parent
 * matrix has no other references (including itself).
 */
void deallocate_matrix(matrix *mat) {
    // Task 1.3 TODO
    // HINTS: Follow these steps.
    // 1. If the matrix pointer `mat` is NULL, return.
    // 2. If `mat` has no parent: decrement its `ref_cnt` field by 1. If the `ref_cnt` field becomes 0, then free `mat` and its `data` field.
    // 3. Otherwise, recursively call `deallocate_matrix` on `mat`'s parent, then free `mat`.
    if (mat == NULL) {
        return;
    }
    if (mat->parent == NULL) {
        mat->ref_cnt = mat->ref_cnt - 1;
        if (mat->ref_cnt == 0) {
            free(mat->data);
            free(mat);
        }
    } else {
        deallocate_matrix(mat->parent);
        free(mat);
    }

    return;
}

/*
 * Allocates space for a matrix struct pointed to by `mat` with `rows` rows and `cols` columns.
 * Its data should point to the `offset`th entry of `from`'s data (you do not need to allocate memory)
 * for the data field. `parent` should be set to `from` to indicate this matrix is a slice of `from`
 * and the reference counter for `from` should be incremented. Lastly, do not forget to set the
 * matrix's row and column values as well.
 * You should return -1 if either `rows` or `cols` or both have invalid values. Return -2 if any
 * call to allocate memory in this function fails.
 * Return 0 upon success.
 * NOTE: Here we're allocating a matrix struct that refers to already allocated data, so
 * there is no need to allocate space for matrix data.
 */
int allocate_matrix_ref(matrix **mat, matrix *from, int offset, int rows, int cols) {
    // Task 1.4 TODO
    // HINTS: Follow these steps.
    // 1. Check if the dimensions are valid. Return -1 if either dimension is not positive.
    // 2. Allocate space for the new matrix struct. Return -2 if allocating memory failed.
    // 3. Set the `data` field of the new struct to be the `data` field of the `from` struct plus `offset`.
    // 4. Set the number of rows and columns in the new struct according to the arguments provided.
    // 5. Set the `parent` field of the new struct to the `from` struct pointer.
    // 6. Increment the `ref_cnt` field of the `from` struct by 1.
    // 7. Store the address of the allocated matrix struct at the location `mat` is pointing at.
    // 8. Return 0 upon success.
    if (rows < 1 || cols < 1) {
        return -1; 
    } 
    struct matrix *child = malloc(sizeof(matrix));

    if (child == NULL) {
        return -2;
    }

    child->data = from->data + offset; 
    child->rows = rows; 
    child->cols = cols; 
    child->parent = from; 
    from->ref_cnt = from->ref_cnt + 1; 
    *mat = child; 
    return 0; 
}

/*
 * Sets all entries in mat to val. Note that the matrix is in row-major order.
 */
void fill_matrix(matrix *mat, double val) {
    // Task 1.5 TODO

    int rows = mat->rows;
    int cols = mat->cols; 
    int size = rows * cols;
    double *data = mat->data;

    __m256d vector = _mm256_set1_pd(val);
    #pragma omp parallel for
    for (int i = 0; i < size/16 * 16; i+= 16) {
        _mm256_storeu_pd((double *) (data + i), vector);
        _mm256_storeu_pd((double *) (data + i + 4), vector);
        _mm256_storeu_pd((double *) (data + i + 8), vector);
        _mm256_storeu_pd((double *) (data + i + 12), vector);
    }

    //tail case 
    #pragma omp parallel for
    for (int i = size/16 * 16; i < size; i++) {
        data[i] = val;
    }
}

/*
 * Store the result of taking the absolute value element-wise to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int abs_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    //_mm_abs_epi8
    double *data = mat->data; 
    double *data_res = result->data;
    int size = mat->rows * mat->cols;

    // for (int i = 0; i < size/4 * 4; i+= 4) {
    //     data_res[i] = data[i] > 0? data[i]: -data[i];
    //     data_res[i + 1] = data[i + 1] > 0? data[i + 1]: -data[i + 1];
    //     data_res[i + 2] = data[i + 2] > 0? data[i + 2]: -data[i + 2];
    //     data_res[i + 3] = data[i + 3] > 0? data[i + 3]: -data[i + 3];
    // }
    __m256d neg = _mm256_set1_pd(-1);
    # pragma omp parallel for 
    for (int i = 0; i < size/ 8 * 8; i+= 8) {
        __m256d load_data1 = _mm256_loadu_pd((double *) (data + i)); //loads the first 4 elements of data
        __m256d load_data2 = _mm256_loadu_pd((double *) (data + i + 4)); //loads the next 4 elements of data
        __m256d opp1 = _mm256_mul_pd(neg, load_data1);
        __m256d opp2 = _mm256_mul_pd(neg, load_data2);
        __m256d max1 = _mm256_max_pd(opp1, load_data1);
        __m256d max2 = _mm256_max_pd(opp2, load_data2);
        _mm256_storeu_pd((double *) (data_res + i), max1);
        _mm256_storeu_pd((double *) (data_res + i + 4 ), max2);
    }
    //# pragma omp parallel for 
    for (int i = size/8 * 8; i < size; i++) {
        data_res[i] = data[i] > 0? data[i]: -data[i];
    }
    
    return 0;

}

/*
 * (OPTIONAL)
 * Store the result of element-wise negating mat's entries to `result`.
 * Return 0 upon success.
 * Note that the matrix is in row-major order.
 */
int neg_matrix(matrix *result, matrix *mat) {
    // Task 1.5 TODO
    double *data = mat->data;
    double *data_res = result->data;
    int size = mat->rows * mat->cols;

    __m256d zeroes = _mm256_setzero_pd();

    #pragma omp parallel for
    for (int i = 0; i < size/8 * 8; i+= 8) {
        __m256d load_data1 = _mm256_loadu_pd((double *) (data + i)); //loads the first 4 elements of data
        __m256d load_data2 = _mm256_loadu_pd((double *) (data + i + 4)); //loads the next 4 elements of data
        __m256d neg1 = _mm256_sub_pd(zeroes, load_data1);
        __m256d neg2 = _mm256_sub_pd(zeroes, load_data2);
        _mm256_storeu_pd((double *) (data_res + i), neg1);
        _mm256_storeu_pd((double *) (data_res + i + 4), neg2);
    }

    for (int i = size/8 * 8; i < size; i++) {
        data_res[i] = -data[i];
    }

    return 0;
}

/*
 * Store the result of adding mat1 and mat2 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int add_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int dims = mat1->rows * mat1->cols;
    double *data1 = mat1->data;
    double *data2 = mat2->data;
    double *res = result->data;

    // #pragma omp parallel for
    // for (int i = 0; i < dims/4 * 4; i+= 4) {
    //     res[i] = data1[i] + data2[i];
    //     res[i + 1] = data1[i + 1] + data2[i + 1];
    //     res[i + 2] = data1[i + 2] + data2[i + 2];
    //     res[i + 3] = data1[i + 3] + data2[i + 3];
    // }

    // //tail case
    // for (int i = dims/4 * 4; i < dims; i++) {
    //     res[i] = data1[i] + data2[i];
    // }
    # pragma omp parallel for 
    for (unsigned int i = 0; i < dims/8 * 8; i+= 8) {
        __m256d load_data1 = _mm256_loadu_pd((double *) (data1 + i)); //loads the first 4 elements of data1
        __m256d load_data2 = _mm256_loadu_pd((double *) (data2 + i)); //loads the first 4 elements of data2
        __m256d load_data1_2 = _mm256_loadu_pd((double *) (data1 + i + 4)); //loads the next 4 elements of data1
        __m256d load_data2_2 = _mm256_loadu_pd((double *) (data2 + i + 4)); //loads the next 4 elements of data2
        __m256d added = _mm256_add_pd(load_data1, load_data2);
        __m256d added_2 = _mm256_add_pd(load_data1_2, load_data2_2);
        _mm256_storeu_pd((double *) (res + i), added);
        _mm256_storeu_pd((double *) (res + i + 4), added_2);

    }

        // //tail case
    // #pragma omp parallel for 
    for (int i = dims/8 * 8; i < dims; i++) {
        res[i] = data1[i] + data2[i];
    }

    return 0;

    
}

/*
 * (OPTIONAL)
 * Store the result of subtracting mat2 from mat1 to `result`.
 * Return 0 upon success.
 * You may assume `mat1` and `mat2` have the same dimensions.
 * Note that the matrix is in row-major order.
 */
int sub_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.5 TODO
    int dims = mat1->rows * mat1->cols;
    double *data1 = mat1->data;
    double *data2 = mat2->data;
    double *res = result->data;

    #pragma omp parallel for
    for (int i = 0; i < dims/8 * 8; i+= 8) {
        __m256d load_data1 = _mm256_loadu_pd((double *) (data1 + i)); //loads the first 4 elements of data1
        __m256d load_data2 = _mm256_loadu_pd((double *) (data2 + i)); //loads the first 4 elements of data2
        __m256d load_data1_2 = _mm256_loadu_pd((double *) (data1 + i + 4)); //loads the next 4 elements of data1
        __m256d load_data2_2 = _mm256_loadu_pd((double *) (data2 + i + 4)); //loads the next 4 elements of data2
        __m256d subbed = _mm256_sub_pd(load_data1, load_data2);
        __m256d subbed_2 = _mm256_sub_pd(load_data1_2, load_data2_2);
        _mm256_storeu_pd((double *) (res + i), subbed);
        _mm256_storeu_pd((double *) (res + i + 4), subbed_2);
    }

    //tail case
    for (int i = dims/8 * 8; i < dims; i++) {
        res[i] = data1[i] - data2[i];
    }
    return 0;
}

/*
* Transposes matrix mat2. Returns the computation of transposing matrix mat2 to matrix result.
* Return 0 upon success.
*/
int transpose_matrix(matrix *result, matrix *mat2) {
    // 3x2 mat2 -> 2x3 result 
    //result is allocated outside of transpose_matrix function call inside mul_matrix
    int rowL = mat2->rows;
    int colL = mat2->cols;
    //read the columns of matrix mat2
    //the column becomes the row in result
    //
    #pragma omp parallel for 
    for (int col = 0; col < colL; col++) {
        for (int row = 0; row < rowL; row++) {
            result->data[col * rowL + row] = mat2->data[col + row * colL]; //column major order
        }
    }

    return 0;
}

/*
 * Store the result of multiplying mat1 and mat2 to `result`.
 * Return 0 upon success.
 * Remember that matrix multiplication is not the same as multiplying individual elements.
 * You may assume `mat1`'s number of columns is equal to `mat2`'s number of rows.
 * Note that the matrix is in row-major order.
 */
int mul_matrix(matrix *result, matrix *mat1, matrix *mat2) {
    // Task 1.6 TODO
    // int rowR = mat1->rows;
    // int colR = mat2->cols;
    // double data1;
    // double data2;
    // double curr;

    // for (int i = 0; i < rowR; i++) {
    //     for (int j = 0; j < colR; j++) {
    //         set(result, i, j, 0);
    //         for (int k = 0; k < mat1->cols; k++) {
    //             data1 = get(mat1, i, k);
    //             data2 = get(mat2, k, j); 
    //             curr = get(result, i, j);
    //             set(result, i, j, curr + (data1 * data2));
    //         }
    //     }
    // }

    matrix *transpose;
    allocate_matrix(&transpose, mat2->cols, mat2->rows);
    transpose_matrix(transpose, mat2); 
    // CACHE BLOCKING SIZE 32 
    int BLOCK_SIZE = 32; 
    
    #pragma omp parallel for
    for (int row = 0; row < mat1->rows; i+= BLOCK_SIZE) {
        for (int col = 0; col < mat2->cols; j += BLOCK_SIZE) {
            for (int BLOCK_ROW = row; BLOCK_ROW < row + BLOCK_SIZE; BLOCK_ROW++) {
                for (int BLOCK_COL = col; BLOCK_COL < col + BLOCK_SIZE; BLOCK_COL++) {
                    // IF WE ARE STILL IN THE SAME ROW AND COL 
                    if (BLOCK_ROW < mat1->rows && BLOCK_COL < mat2->cols) {
                        __m256d mat1_input; 
                        __m256d mat2T_input;
                        result->data[BLOCK_ROW * result->cols + BLOCK_COL] = 0;
                        __m256d partial_row_sum = _mm256_set1_pd(0);
                        for (int i = 0; i < (mat1->cols / 16) * 16; i+= 16) {
                            // OUR MATRIX HAS DIM >= 16 X 16
                            // Zeroth 4 
                            mat1_input = __m256_loadu_((mat1->data + BLOCK_ROW * mat1->cols + i));
                            mat2T_input = __m256_loadu_((transpose->data + BLOCK_COL * transpose->cols + i ));
                            partial_row_sum = _mm256_fmadd_pd(mat1_input, mat2T_input, partial_row_sum);
                            // First 4 
                            mat1_input = __m256_loadu_((mat1->data + BLOCK_ROW * mat1->cols + i + 4));
                            mat2T_input = __m256_loadu_((transpose->data + BLOCK_COL * transpose->cols + i + 4));
                            partial_row_sum = _mm256_fmadd_pd(mat1_input, mat2T_input, partial_row_sum);
                            // Second 4 
                            mat1_input = __m256_loadu_((mat1->data + BLOCK_ROW * mat1->cols + i + 8));
                            mat2T_input = __m256_loadu_((transpose->data + BLOCK_COL * transpose->cols + i + 8));
                            partial_row_sum = _mm256_fmadd_pd(mat1_input, mat2T_input, partial_row_sum);
                            // Third 4 
                            mat1_input = __m256_loadu_((mat1->data + BLOCK_ROW * mat1->cols + i + 12));
                            mat2T_input = __m256_loadu_((transpose->data + BLOCK_COL * transpose->cols + i + 12));
                            partial_row_sum = _mm256_fmadd_pd(mat1_input, mat2T_input, partial_row_sum);
                        }
                        double temp[4];
                        _mm256_storeu_pd(temp, partial_row_sum);
                        // TAIL CASE FOR MATRIX WITH DIM < 16 X 16 OR REMAININGS OF EACH ROW 
                        for (int i = (mat1->cols / 16) * 16; i < mat1->cols; i++) {
                            // We are just adding the results together so doesn't matter what index is added
                            temp[0] += mat1->data[BLOCK_ROW * mat1->cols + i] * transpose->data[BLOCK_COL * transpose->cols + i];
                        }
                        // STORING PARTIAL SUM AND TAIL CASE SUM INTO THAT ONE ELEMENT OF RESULT
                        result->data[BLOCK_ROW * result->cols + BLOCK_COL] = temp[0] + temp[1] + temp[2] + temp[3];
                    }
                }
            }
        }
    }





    deallocate_matrix(transpose);

    return 0;
}


/* 
* Stores the identity matrix in result.
*/
int np_eye(matrix *result) {

    #pragma omp parallel for 
    for (int i = 0; i < result->rows; i++) {
        result->data[i + i * result->cols] = 1.0;
    }

    return 0;
}



/*
 * Store the result of raising mat to the (pow)th power to `result`.
 * Return 0 upon success.
 * Remember that pow is defined with matrix multiplication, not element-wise multiplication.
 * You may assume `mat` is a square matrix and `pow` is a non-negative integer.
 * Note that the matrix is in row-major order.
 */
int pow_matrix(matrix *result, matrix *mat, int pow) {
    // Task 1.6 TODO
    int dims = result->cols * result->cols;
    if (pow == 0) {
        np_eye(result);
        return 0;
    } else if (pow == 1) {
        memcpy(result->data, mat->data, dims * sizeof(double));
        return 0;
    } else {

    
        matrix *temp;
        matrix *temp2;
        allocate_matrix(&temp, result->rows, result->cols);
        allocate_matrix(&temp2, result->rows, result->cols);
        if (temp == NULL) {
            return -1;
        }
        if (temp2 == NULL) {
            return -1;
        }

        mul_matrix(temp, mat, mat);
        pow_matrix(temp2, temp, pow / 2);
        if (pow % 2 == 1) {
            mul_matrix(result, temp2, mat);
        } else {
            memcpy(result->data, temp2->data, dims * sizeof(double));
        }

        deallocate_matrix(temp);
        deallocate_matrix(temp2);
        return 0;
    }
}
