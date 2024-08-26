#define MT_IMPLEMENTATION
#include "../mint.h"

#define SZ_F sizeof(mt_float)
#define SZ_I sizeof(int)

#define MT_ASSERT_TEST_F(test_name, condition, format, ...)                    \
    do {                                                                       \
        fprintf(stderr, test_name ": ");                                       \
        if (!(condition)) {                                                    \
            stats->failed++;                                                   \
            fprintf(stderr, "\x1b[31m");                                       \
            fprintf(stderr, "FAILED [%s:%d]: %s\n  ", __FILE__, __LINE__,      \
                    #condition);                                               \
            fprintf(stderr, format, __VA_ARGS__);                              \
            fprintf(stderr, "\n");                                             \
            fprintf(stderr, "\x1b[0m");                                        \
        } else {                                                               \
            stats->pass++;                                                     \
            fprintf(stderr, "\e[0;32m"                                         \
                            "PASS\n"                                           \
                            "\x1b[0m");                                        \
        }                                                                      \
    } while (0)

#define MT_ASSERT_TEST(test_name, condition)                                   \
    MT_ASSERT_TEST_F(test_name, condition, "%s", "")

// clang-format off

typedef struct {
    int failed;
    int pass;
} Stats;

static inline int mt_arr_same(const void *arr1, const void *arr2, int len,
                              size_t size) {
    const char *p1 = arr1;
    const char *p2 = arr2;

    for (int i = 0; i < len; i++) {
        if (memcmp(p1 + i * size, p2 + i * size, size) != 0) {
            return 0;
        }
    }
    return 1;
}

#define MT_TENSOR_BULK_FREE(len, ...)                                          \
    do {                                                                       \
        for (int i = 0; i < len; ++i)                                          \
            mt_tensor_free((mt_tensor *[]){__VA_ARGS__}[i]);                   \
    } while (0)

#define MT_SECTION_TITLE(msg)                                                  \
    printf("\e[1;33m"                                                          \
           "\n" msg "\e[0m \n");

void test_binop(Stats *stats) {
    MT_SECTION_TITLE("Simple binops");
    mt_tensor *a = mt_tensor_alloc_values(MT_ARR_INT(2), 1, MT_ARR_FLOAT(1, 2));
    mt_tensor *b = mt_tensor_alloc_values(MT_ARR_INT(2), 1, MT_ARR_FLOAT(2, 4));

    mt_tensor *res_add = mt_add(a, b);
    mt_tensor *res_sub = mt_sub(a, b);
    mt_tensor *res_mul = mt_mul(a, b);
    mt_tensor *res_div = mt_div(a, b);

    MT_ASSERT_TEST("simple add",
                   mt_arr_same(res_add->data, MT_ARR_FLOAT(3, 6), 2, SZ_F));
    MT_ASSERT_TEST("simple sub",
                   mt_arr_same(res_sub->data, MT_ARR_FLOAT(-1, -2), 2, SZ_F));
    MT_ASSERT_TEST("simple mul",
                   mt_arr_same(res_mul->data, MT_ARR_FLOAT(2, 8), 2, SZ_F));
    MT_ASSERT_TEST("simple div",
                   mt_arr_same(res_div->data, MT_ARR_FLOAT(0.5, 0.5), 2, SZ_F));

    MT_TENSOR_BULK_FREE(6, a, b, res_add, res_sub, res_mul, res_div);

    MT_SECTION_TITLE("broadcast binop (2, 2) (2)");
    a = mt_tensor_alloc_values(MT_ARR_INT(2, 2), 2, MT_ARR_FLOAT(1, 2, 3, 4));
    b = mt_tensor_alloc_values(MT_ARR_INT(2), 1, MT_ARR_FLOAT(1, 2));
    res_add = mt_add(a, b);
    res_mul = mt_mul(a, b);
    MT_ASSERT_TEST("add ndim", res_add->ndim == 2);
    MT_ASSERT_TEST("add shape",
                   mt_arr_same(res_add->shape, MT_ARR_INT(2, 2), 2, SZ_I));
    MT_ASSERT_TEST("add data", mt_arr_same(res_add->data,
                                           MT_ARR_FLOAT(2, 4, 4, 6), 4, SZ_F));
    MT_ASSERT_TEST("mul ndim", res_mul->ndim == 2);
    MT_ASSERT_TEST("mul shape",
                   mt_arr_same(res_mul->shape, MT_ARR_INT(2, 2), 2, SZ_I));
    MT_ASSERT_TEST("mul data", mt_arr_same(res_mul->data,
                                           MT_ARR_FLOAT(1, 4, 3, 8), 4, SZ_F));
    MT_TENSOR_BULK_FREE(4, a, b, res_add, res_mul);

    //
    // If both above pass, we assume other simple operator will also pass
    //

    MT_SECTION_TITLE("broadcast binop (3, 1) (1, 3)");
    a = mt_tensor_alloc_values(MT_ARR_INT(3, 1), 2, MT_ARR_FLOAT(1, 2, 3));
    b = mt_tensor_alloc_values(MT_ARR_INT(1, 3), 2, MT_ARR_FLOAT(1, 2, 3));
    res_add = mt_add(a, b);
    res_mul = mt_mul(a, b);
    MT_ASSERT_TEST("add ndim", res_add->ndim == 2);
    MT_ASSERT_TEST("add shape",
                   mt_arr_same(res_add->shape, MT_ARR_INT(3, 3), 2, SZ_I));
    MT_ASSERT_TEST("add data",
                   mt_arr_same(res_add->data,
                               MT_ARR_FLOAT(2, 3, 4, 3, 4, 5, 4, 5, 6), 4,
                               SZ_F));
    MT_ASSERT_TEST("mul ndim", res_mul->ndim == 2);
    MT_ASSERT_TEST("mul shape",
                   mt_arr_same(res_mul->shape, MT_ARR_INT(3, 3), 2, SZ_I));
    MT_ASSERT_TEST("mul data",
                   mt_arr_same(res_mul->data,
                               MT_ARR_FLOAT(1, 2, 3, 2, 4, 6, 3, 6, 9), 4,
                               SZ_F));
    MT_TENSOR_BULK_FREE(4, a, b, res_add, res_mul);
}

void test_permute_dim(Stats *stats) {
    MT_SECTION_TITLE("(0,1)->(1,0)");
    mt_tensor *a   = mt_tensor_alloc_values(MT_ARR_INT(3, 2), 2,
                                            MT_ARR_FLOAT(1, 2, 3, 4, 5, 6));
    mt_tensor *res = mt_tensor_permute_dims(a, MT_ARR_INT(1, 0));
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(2, 3), 2, SZ_I));
    MT_ASSERT_TEST(
        "data",
        mt_arr_same(res->data, MT_ARR_FLOAT(1, 3, 5, 2, 4, 6), 6, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(res);

    MT_SECTION_TITLE("(c,h,w)->(w,h,c)");
    a   = mt_tensor_alloc_values(MT_ARR_INT(3, 2, 4), 3,
                                 MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                              12, 13, 14, 15, 16, 17, 18, 19, 20,
                                              21, 22, 23, 24));
    res = mt_tensor_permute_dims(a, MT_ARR_INT(2, 1, 0));
    MT_ASSERT_TEST("shape",
                   mt_arr_same(res->shape, MT_ARR_INT(4, 2, 3), 3, SZ_I));
    MT_ASSERT_TEST(
        "data",
        mt_arr_same(res->data,
                    MT_ARR_FLOAT(1, 9, 17, 5, 13, 21, 2, 10, 18, 6, 14, 22, 3,
                                 11, 19, 7, 15, 23, 4, 12, 20, 8, 16, 24),
                    24, SZ_F));

    mt_tensor_free(a);
    mt_tensor_free(res);

    // clang-format off
    MT_SECTION_TITLE("5D Permutation: (a,b,c,d,e)->(e,d,c,b,a)");

    // Create a 5D tensor with dimensions 2x2x2x2x2
    a = mt_tensor_alloc_values(
        MT_ARR_INT(2, 2, 2, 2, 2), 5,
        MT_ARR_FLOAT(
                1,  2,  3,  4,  5,  6,  7,  8,
                9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32
        )
    );

    // Perform the permutation
    res = mt_tensor_permute_dims(a, MT_ARR_INT(4, 3, 2, 1, 0));

    // Check the shape
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(2, 2, 2, 2, 2), 5, SZ_I));

    // Check the data
    MT_ASSERT_TEST(
        "data",
        mt_arr_same(res->data,
            MT_ARR_FLOAT(
                    1, 17,  9, 25,  5, 21, 13, 29,
                    3, 19, 11, 27,  7, 23, 15, 31,
                    2, 18, 10, 26,  6, 22, 14, 30,
                    4, 20, 12, 28,  8, 24, 16, 32
            ),
            32, SZ_F)
    );

    mt_tensor_free(a);
    mt_tensor_free(res);
    // clang-format on
}

void test_slice(Stats *stats) {
    MT_SECTION_TITLE("Basic 2D slice");
    mt_tensor *a = mt_tensor_alloc_values(
        MT_ARR_INT(4, 4), 2,
        MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    int        starts[] = {1, 1};
    int        ends[]   = {3, 3};
    int        axes[]   = {0, 1};
    int        steps[]  = {1, 1};
    mt_tensor *res      = mt_tensor_slice(a, starts, ends, axes, steps, 2);
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(2, 2), 2, SZ_I));
    MT_ASSERT_TEST("data",
                   mt_arr_same(res->data, MT_ARR_FLOAT(6, 7, 10, 11), 4, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(res);

    MT_SECTION_TITLE("3D slice with negative indices");
    a = mt_tensor_alloc_values(
        MT_ARR_INT(3, 4, 5), 3,
        MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                     33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                     48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60));
    int        starts2[] = {1, 0, -2};
    int        ends2[]   = {3, -1, 5};
    int        axes2[]   = {0, 1, 2};
    int        steps2[]  = {1, 1, 1};
    mt_tensor *res2      = mt_tensor_slice(a, starts2, ends2, axes2, steps2, 3);
    MT_ASSERT_TEST("shape",
                   mt_arr_same(res2->shape, MT_ARR_INT(2, 3, 2), 3, SZ_I));
    MT_ASSERT_TEST("data", mt_arr_same(res2->data,
                                       MT_ARR_FLOAT(24, 25, 29, 30, 34, 35, 44,
                                                    45, 49, 50, 54, 55),
                                       12, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(res2);

    MT_SECTION_TITLE("2D slice with steps");
    mt_tensor *b = mt_tensor_alloc_values(
        MT_ARR_INT(5, 5), 2,
        MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                     18, 19, 20, 21, 22, 23, 24, 25));
    int        starts3[] = {0, 0};
    int        ends3[]   = {5, 5};
    int        steps3[]  = {2, 2};
    mt_tensor *res3      = mt_tensor_slice(b, starts3, ends3, NULL, steps3, 2);
    MT_ASSERT_TEST("shape",
                   mt_arr_same(res3->shape, MT_ARR_INT(3, 3), 2, SZ_I));
    MT_ASSERT_TEST("data",
                   mt_arr_same(res3->data,
                               MT_ARR_FLOAT(1, 3, 5, 11, 13, 15, 21, 23, 25), 9,
                               SZ_F));
    mt_tensor_free(res3);
}

void test_concat(Stats *stats) {
    MT_SECTION_TITLE("Basic 2D concat along axis 0");
    mt_tensor *a        = mt_tensor_alloc_values(MT_ARR_INT(2, 3), 2,
                                                 MT_ARR_FLOAT(1, 2, 3, 4, 5, 6));
    mt_tensor *b        = mt_tensor_alloc_values(MT_ARR_INT(2, 3), 2,
                                                 MT_ARR_FLOAT(7, 8, 9, 10, 11, 12));
    mt_tensor *inputs[] = {a, b};
    mt_tensor *res      = mt_concat(inputs, 2, 0);
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(4, 3), 2, SZ_I));
    MT_ASSERT_TEST(
        "data", mt_arr_same(res->data,
                            MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                            12, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(res);

    MT_SECTION_TITLE("2D concat along axis 1");
    a         = mt_tensor_alloc_values(MT_ARR_INT(3, 2), 2,
                                       MT_ARR_FLOAT(1, 2, 3, 4, 5, 6));
    b         = mt_tensor_alloc_values(MT_ARR_INT(3, 2), 2,
                                       MT_ARR_FLOAT(7, 8, 9, 10, 11, 12));
    inputs[0] = a;
    inputs[1] = b;
    res       = mt_concat(inputs, 2, 1);
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(3, 4), 2, SZ_I));
    MT_ASSERT_TEST(
        "data", mt_arr_same(res->data,
                            MT_ARR_FLOAT(1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12),
                            12, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(res);

    MT_SECTION_TITLE("3D concat along axis 1");
    a         = mt_tensor_alloc_values(MT_ARR_INT(2, 2, 2), 3,
                                       MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8));
    b         = mt_tensor_alloc_values(MT_ARR_INT(2, 1, 2), 3,
                                       MT_ARR_FLOAT(9, 10, 11, 12));
    inputs[0] = a;
    inputs[1] = b;
    res       = mt_concat(inputs, 2, 1);
    MT_ASSERT_TEST("shape",
                   mt_arr_same(res->shape, MT_ARR_INT(2, 3, 2), 3, SZ_I));
    MT_ASSERT_TEST(
        "data", mt_arr_same(res->data,
                            MT_ARR_FLOAT(1, 2, 3, 4, 9, 10, 5, 6, 7, 8, 11, 12),
                            12, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(res);

    MT_SECTION_TITLE("1D concat");
    a = mt_tensor_alloc_values(MT_ARR_INT(3), 1, MT_ARR_FLOAT(1, 2, 3));
    b = mt_tensor_alloc_values(MT_ARR_INT(2), 1, MT_ARR_FLOAT(4, 5));
    mt_tensor *c = mt_tensor_alloc_values(MT_ARR_INT(1), 1, MT_ARR_FLOAT(6));
    mt_tensor *inputs_1d[] = {a, b, c};
    res                    = mt_concat(inputs_1d, 3, 0);
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(6), 1, SZ_I));
    MT_ASSERT_TEST(
        "data",
        mt_arr_same(res->data, MT_ARR_FLOAT(1, 2, 3, 4, 5, 6), 6, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(c);
    mt_tensor_free(res);

    MT_SECTION_TITLE("4D concat along axis 2");
    a = mt_tensor_alloc_values(
        MT_ARR_INT(2, 2, 2, 2), 4,
        MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16));
    b         = mt_tensor_alloc_values(MT_ARR_INT(2, 2, 1, 2), 4,
                                       MT_ARR_FLOAT(17, 18, 19, 20, 21, 22, 23, 24));
    inputs[0] = a;
    inputs[1] = b;
    res       = mt_concat(inputs, 2, 2);
    MT_ASSERT_TEST("shape",
                   mt_arr_same(res->shape, MT_ARR_INT(2, 2, 3, 2), 4, SZ_I));
    MT_ASSERT_TEST(
        "data",
        mt_arr_same(res->data,
                    MT_ARR_FLOAT(1, 2, 3, 4, 17, 18, 5, 6, 7, 8, 19, 20, 9, 10,
                                 11, 12, 21, 22, 13, 14, 15, 16, 23, 24),
                    24, SZ_F));

    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(res);

    MT_SECTION_TITLE("2D concat with negative axis");
    a         = mt_tensor_alloc_values(MT_ARR_INT(2, 3), 2,
                                       MT_ARR_FLOAT(1, 2, 3, 4, 5, 6));
    b         = mt_tensor_alloc_values(MT_ARR_INT(2, 3), 2,
                                       MT_ARR_FLOAT(7, 8, 9, 10, 11, 12));
    inputs[0] = a;
    inputs[1] = b;
    res       = mt_concat(inputs, 2, -2); // Equivalent to axis 0
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(4, 3), 2, SZ_I));
    MT_ASSERT_TEST(
        "data", mt_arr_same(res->data,
                            MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                            12, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(res);

    MT_SECTION_TITLE("2D concat with negative axis");
    a         = mt_tensor_alloc_values(MT_ARR_INT(2, 3), 2,
                                       MT_ARR_FLOAT(1, 2, 3, 4, 5, 6));
    b         = mt_tensor_alloc_values(MT_ARR_INT(2, 3), 2,
                                       MT_ARR_FLOAT(7, 8, 9, 10, 11, 12));
    inputs[0] = a;
    inputs[1] = b;
    res       = mt_concat(inputs, 2, -2); // Equivalent to axis 0
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(4, 3), 2, SZ_I));
    MT_ASSERT_TEST(
        "data", mt_arr_same(res->data,
                            MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                            12, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(res);

    MT_SECTION_TITLE("2D concat with negative axis (along last dimension)");
    a = mt_tensor_alloc_values(MT_ARR_INT(3, 2), 2,
                               MT_ARR_FLOAT(1, 2, 3, 4, 5, 6));
    b = mt_tensor_alloc_values(MT_ARR_INT(3, 1), 2, MT_ARR_FLOAT(7, 8, 9));
    inputs[0] = a;
    inputs[1] = b;
    res       = mt_concat(inputs, 2, -1); // Equivalent to axis 1
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(3, 3), 2, SZ_I));
    MT_ASSERT_TEST("data", mt_arr_same(res->data,
                                       MT_ARR_FLOAT(1, 2, 7, 3, 4, 8, 5, 6, 9),
                                       9, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(res);

    MT_SECTION_TITLE("2D concat with varying sizes on concat axis");
    a = mt_tensor_alloc_values(MT_ARR_INT(3, 2), 2,
                               MT_ARR_FLOAT(1, 2, 3, 4, 5, 6));
    b = mt_tensor_alloc_values(MT_ARR_INT(3, 3), 2,
                               MT_ARR_FLOAT(7, 8, 9, 10, 11, 12, 13, 14, 15));
    c = mt_tensor_alloc_values(MT_ARR_INT(3, 1), 2, MT_ARR_FLOAT(16, 17, 18));
    mt_tensor *inputs_varying[] = {a, b, c};
    res = mt_concat(inputs_varying, 3, 1); // Concatenate along axis 1
    MT_ASSERT_TEST("shape", mt_arr_same(res->shape, MT_ARR_INT(3, 6), 2, SZ_I));
    MT_ASSERT_TEST("data",
                   mt_arr_same(res->data,
                               MT_ARR_FLOAT(1, 2, 7, 8, 9, 16, 3, 4, 10, 11, 12,
                                            17, 5, 6, 13, 14, 15, 18),
                               18, SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(b);
    mt_tensor_free(c);
    mt_tensor_free(res);
}
void test_tensor_pad(Stats *stats) {
    {
        MT_SECTION_TITLE("2D tensor reflect pad");
        mt_tensor *input = mt_tensor_alloc_values(
            MT_ARR_INT(3, 3), 2, MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9));
        int        pads[] = {1, 1, 1, 1};
        mt_tensor *output = mt_tensor_pad(input, pads, MT_PAD_REFLECT, 0);
        // Verify shape
        MT_ASSERT_TEST("shape",
                       mt_arr_same(output->shape, MT_ARR_INT(5, 5), 2, SZ_I));

        // Verify data
        mt_float expected[] = {5, 4, 5, 6, 5, 2, 1, 2, 3, 2, 5, 4, 5,
                               6, 5, 8, 7, 8, 9, 8, 5, 4, 5, 6, 5};
        MT_ASSERT_TEST("data", mt_arr_same(output->data, expected, 25, SZ_F));

        mt_tensor_free(input);
        mt_tensor_free(output);
    }

    MT_SECTION_TITLE("1D tensor reflect pad");
    mt_tensor *a =
        mt_tensor_alloc_values(MT_ARR_INT(5), 1, MT_ARR_FLOAT(1, 2, 3, 4, 5));
    int        pads_1d[] = {2, 2};
    mt_tensor *res_1d    = mt_tensor_pad(a, pads_1d, MT_PAD_REFLECT, 0);
    MT_ASSERT_TEST("1D shape",
                   mt_arr_same(res_1d->shape, MT_ARR_INT(9), 1, SZ_I));
    MT_ASSERT_TEST("1D data",
                   mt_arr_same(res_1d->data,
                               MT_ARR_FLOAT(3, 2, 1, 2, 3, 4, 5, 4, 3), 9,
                               SZ_F));
    mt_tensor_free(a);
    mt_tensor_free(res_1d);

    // 1D tensor reflect pad
    {
        MT_SECTION_TITLE("1D tensor reflect pad");
        mt_tensor *a         = mt_tensor_alloc_values(MT_ARR_INT(5), 1,
                                                      MT_ARR_FLOAT(1, 2, 3, 4, 5));
        int        pads_1d[] = {2, 2};
        mt_tensor *res_1d    = mt_tensor_pad(a, pads_1d, MT_PAD_REFLECT, 0);
        MT_ASSERT_TEST("1D shape",
                       mt_arr_same(res_1d->shape, MT_ARR_INT(9), 1, SZ_I));
        MT_ASSERT_TEST("1D data",
                       mt_arr_same(res_1d->data,
                                   MT_ARR_FLOAT(3, 2, 1, 2, 3, 4, 5, 4, 3), 9,
                                   SZ_F));
        mt_tensor_free(a);
        mt_tensor_free(res_1d);
    }

    // 1D tensor reflect pad with asymmetric padding
    MT_SECTION_TITLE("1D tensor reflect pad with asymmetric padding");
    {
        mt_tensor *a              = mt_tensor_alloc_values(MT_ARR_INT(5), 1,
                                                           MT_ARR_FLOAT(1, 2, 3, 4, 5));
        int        pads_1d_asym[] = {1, 3};
        mt_tensor *res_1d_asym =
            mt_tensor_pad(a, pads_1d_asym, MT_PAD_REFLECT, 0);
        MT_ASSERT_TEST("1D asymmetric shape",
                       mt_arr_same(res_1d_asym->shape, MT_ARR_INT(9), 1, SZ_I));
        MT_ASSERT_TEST("1D asymmetric data",
                       mt_arr_same(res_1d_asym->data,
                                   MT_ARR_FLOAT(2, 1, 2, 3, 4, 5, 4, 3, 2), 9,
                                   SZ_F));
        mt_tensor_free(a);
        mt_tensor_free(res_1d_asym);
    }

    // 2D tensor reflect pad
    {
        MT_SECTION_TITLE("2D tensor reflect pad");
        mt_tensor *input = mt_tensor_alloc_values(
            MT_ARR_INT(3, 3), 2, MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9));
        int        pads[] = {1, 1, 1, 1};
        mt_tensor *output = mt_tensor_pad(input, pads, MT_PAD_REFLECT, 0);
        MT_ASSERT_TEST("2D shape",
                       mt_arr_same(output->shape, MT_ARR_INT(5, 5), 2, SZ_I));
        mt_float expected[] = {5, 4, 5, 6, 5, 2, 1, 2, 3, 2, 5, 4, 5,
                               6, 5, 8, 7, 8, 9, 8, 5, 4, 5, 6, 5};
        MT_ASSERT_TEST("2D data",
                       mt_arr_same(output->data, expected, 25, SZ_F));
        mt_tensor_free(input);
        mt_tensor_free(output);
    }

    // 2D tensor reflect pad with asymmetric padding
    {
        MT_SECTION_TITLE("2D tensor reflect pad with asymmetric padding");
        printf("Test case 1: 2D tensor reflect pad with asymmetric padding\n");
        mt_tensor *input = mt_tensor_alloc_values(
            MT_ARR_INT(3, 3), 2, MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9));
        int pads[] = {1, 2, 0, 1};

        printf("Padding: top=%d, left=%d, bottom=%d, right=%d\n\n", pads[0],
               pads[1], pads[2], pads[3]);

        mt_tensor *output = mt_tensor_pad(input, pads, MT_PAD_REFLECT, 0);

        mt_float expected[] = {2, 1, 1, 2, 3, 3, 1, 1, 1, 2, 3, 3,
                               4, 4, 4, 5, 6, 6, 7, 7, 7, 8, 9, 9};

        mt_tensor *expected_tensor =
            mt_tensor_alloc_values(MT_ARR_INT(4, 6), 2, expected);

        mt_tensor_free(input);
        mt_tensor_free(output);
        mt_tensor_free(expected_tensor);
    }

    // 3D tensor reflect pad
    {
        MT_SECTION_TITLE("3D tensor reflect pad");
        mt_tensor *input = mt_tensor_alloc_values(
            MT_ARR_INT(2, 2, 3), 3,
            MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
        int        pads[] = {0, 1, 1, 0, 1, 1};
        mt_tensor *output = mt_tensor_pad(input, pads, MT_PAD_REFLECT, 0);
        MT_ASSERT_TEST("3D shape", mt_arr_same(output->shape,
                                               MT_ARR_INT(2, 4, 5), 3, SZ_I));
        mt_float expected[] = {5,  4,  5,  6,  5,  2, 1, 2, 3, 2,
                               5,  4,  5,  6,  5,  2, 1, 2, 3, 2,
                               11, 10, 11, 12, 11, 8, 7, 8, 9, 8,
                               11, 10, 11, 12, 11, 8, 7, 8, 9, 8};
        MT_ASSERT_TEST("3D data",
                       mt_arr_same(output->data, expected, 40, SZ_F));
        mt_tensor_free(input);
        mt_tensor_free(output);
    }

    // 4D tensor reflect pad (NCHW format)
    {
        MT_SECTION_TITLE("4D tensor reflect pad (NCHW format)");
        mt_tensor *input =
            mt_tensor_alloc_values(MT_ARR_INT(2, 2, 2, 2), 4,
                                   MT_ARR_FLOAT(1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                11, 12, 13, 14, 15, 16));
        int        pads[] = {0, 0, 1, 1, 0, 0, 1, 1};
        mt_tensor *output = mt_tensor_pad(input, pads, MT_PAD_REFLECT, 0);
        MT_ASSERT_TEST(
            "4D shape",
            mt_arr_same(output->shape, MT_ARR_INT(2, 2, 4, 4), 4, SZ_I));
        mt_float expected[] = {
            4,  3,  4,  3,  2,  1,  2,  1,  4,  3,  4,  3,  2,  1,  2,  1,
            8,  7,  8,  7,  6,  5,  6,  5,  8,  7,  8,  7,  6,  5,  6,  5,
            12, 11, 12, 11, 10, 9,  10, 9,  12, 11, 12, 11, 10, 9,  10, 9,
            16, 15, 16, 15, 14, 13, 14, 13, 16, 15, 16, 15, 14, 13, 14, 13};
        MT_ASSERT_TEST("4D data",
                       mt_arr_same(output->data, expected, 64, SZ_F));
        mt_tensor_free(input);
        mt_tensor_free(output);
    }

    // Edge case: 1D tensor with no padding
    MT_SECTION_TITLE("1D tensor with no padding");
    {
        mt_tensor *a         = mt_tensor_alloc_values(MT_ARR_INT(5), 1,
                                                      MT_ARR_FLOAT(1, 2, 3, 4, 5));
        int        pads_1d[] = {0, 0};
        mt_tensor *res_1d    = mt_tensor_pad(a, pads_1d, MT_PAD_REFLECT, 0);
        MT_ASSERT_TEST("1D no padding shape",
                       mt_arr_same(res_1d->shape, MT_ARR_INT(5), 1, SZ_I));
        MT_ASSERT_TEST(
            "1D no padding data",
            mt_arr_same(res_1d->data, MT_ARR_FLOAT(1, 2, 3, 4, 5), 5, SZ_F));
        mt_tensor_free(a);
        mt_tensor_free(res_1d);
    }

    // Edge case: 2D tensor with large padding
    {
        MT_SECTION_TITLE("2D tensor with large padding");
        mt_tensor *input  = mt_tensor_alloc_values(MT_ARR_INT(2, 2), 2,
                                                   MT_ARR_FLOAT(1, 2, 3, 4));
        int        pads[] = {3, 3, 3, 3};

        printf("Padding: top=%d, left=%d, bottom=%d, right=%d\n\n", pads[0],
               pads[1], pads[2], pads[3]);

        mt_tensor *output = mt_tensor_pad(input, pads, MT_PAD_REFLECT, 0);

        mt_float expected[] = {4, 3, 4, 3, 4, 3, 4, 3, 2, 1, 2, 1, 2, 1, 2, 1,
                               4, 3, 4, 3, 4, 3, 4, 3, 2, 1, 2, 1, 2, 1, 2, 1,
                               4, 3, 4, 3, 4, 3, 4, 3, 2, 1, 2, 1, 2, 1, 2, 1,
                               4, 3, 4, 3, 4, 3, 4, 3, 2, 1, 2, 1, 2, 1, 2, 1};

        mt_tensor *expected_tensor =
            mt_tensor_alloc_values(MT_ARR_INT(8, 8), 2, expected);

        mt_tensor_free(input);
        mt_tensor_free(output);
        mt_tensor_free(expected_tensor);
    }
}

// Test function
void test_tensor_split(Stats *stats) {
    // 1D tensor split test
    {
        MT_SECTION_TITLE("1D tensor split");
        int        shape[] = {6};
        mt_tensor *t =
            mt_tensor_alloc_values(shape, 1, (mt_float[]){1, 2, 3, 4, 5, 6});
        int        splits[] = {2, 4};
        mt_tensor *result[2];
        mt_tensor_split(t, 0, splits, 2, result);

        MT_ASSERT_TEST("1D split[0] shape",
                       mt_arr_same(result[0]->shape, (int[]){2}, 1, SZ_I));
        MT_ASSERT_TEST(
            "1D split[0] data",
            mt_arr_same(result[0]->data, (mt_float[]){1, 2}, 2, SZ_F));
        MT_ASSERT_TEST("1D split[1] shape",
                       mt_arr_same(result[1]->shape, (int[]){4}, 1, SZ_I));
        MT_ASSERT_TEST(
            "1D split[1] data",
            mt_arr_same(result[1]->data, (mt_float[]){3, 4, 5, 6}, 4, SZ_F));

        mt_tensor_free(t);
        mt_tensor_free(result[0]);
        mt_tensor_free(result[1]);
    }

    // 2D tensor split test
    {
        MT_SECTION_TITLE("2D tensor split");
        int        shape[] = {3, 4};
        mt_tensor *t       = mt_tensor_alloc_values(
            shape, 2, (mt_float[]){1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        int        splits[] = {1, 2, 1};
        mt_tensor *result[3];
        mt_tensor_split(t, 1, splits, 3, result);

        MT_ASSERT_TEST("2D split[0] shape",
                       mt_arr_same(result[0]->shape, (int[]){3, 1}, 2, SZ_I));
        MT_ASSERT_TEST(
            "2D split[0] data",
            mt_arr_same(result[0]->data, (mt_float[]){1, 5, 9}, 3, SZ_F));
        MT_ASSERT_TEST("2D split[1] shape",
                       mt_arr_same(result[1]->shape, (int[]){3, 2}, 2, SZ_I));
        MT_ASSERT_TEST("2D split[1] data",
                       mt_arr_same(result[1]->data,
                                   (mt_float[]){2, 3, 6, 7, 10, 11}, 6, SZ_F));
        MT_ASSERT_TEST("2D split[2] shape",
                       mt_arr_same(result[2]->shape, (int[]){3, 1}, 2, SZ_I));
        MT_ASSERT_TEST(
            "2D split[2] data",
            mt_arr_same(result[2]->data, (mt_float[]){4, 8, 12}, 3, SZ_F));

        mt_tensor_free(t);
        mt_tensor_free(result[0]);
        mt_tensor_free(result[1]);
        mt_tensor_free(result[2]);
    }

    // 3D tensor split test
    {
        MT_SECTION_TITLE("3D tensor split");
        int        shape[] = {2, 3, 4};
        mt_tensor *t       = mt_tensor_alloc_values(
            shape, 3,
            (mt_float[]){1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,

                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
        int        splits[] = {1, 2};
        mt_tensor *result[2];
        mt_tensor_split(t, 1, splits, 2, result);

        MT_ASSERT_TEST(
            "3D split[0] shape",
            mt_arr_same(result[0]->shape, (int[]){2, 1, 4}, 3, SZ_I));
        MT_ASSERT_TEST("3D split[0] data",
                       mt_arr_same(result[0]->data,
                                   (mt_float[]){1, 2, 3, 4, 13, 14, 15, 16}, 8,
                                   SZ_F));
        MT_ASSERT_TEST(
            "3D split[1] shape",
            mt_arr_same(result[1]->shape, (int[]){2, 2, 4}, 3, SZ_I));
        MT_ASSERT_TEST("3D split[1] data",
                       mt_arr_same(result[1]->data,
                                   (mt_float[]){5, 6, 7, 8, 9, 10, 11, 12, 17,
                                                18, 19, 20, 21, 22, 23, 24},
                                   16, SZ_F));

        mt_tensor_free(t);
        mt_tensor_free(result[0]);
        mt_tensor_free(result[1]);
    }

    // 4D tensor split test
    {
        MT_SECTION_TITLE("4D tensor split");
        int        shape[] = {2, 2, 2, 3};
        mt_tensor *t       = mt_tensor_alloc_values(
            shape, 4,
            (mt_float[]){1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,

                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24});
        int        splits[] = {1, 2};
        mt_tensor *result[2];
        mt_tensor_split(t, 3, splits, 2, result);

        MT_ASSERT_TEST(
            "4D split[0] shape",
            mt_arr_same(result[0]->shape, (int[]){2, 2, 2, 1}, 4, SZ_I));
        MT_ASSERT_TEST("4D split[0] data",
                       mt_arr_same(result[0]->data,
                                   (mt_float[]){1, 4, 7, 10, 13, 16, 19, 22}, 8,
                                   SZ_F));
        MT_ASSERT_TEST(
            "4D split[1] shape",
            mt_arr_same(result[1]->shape, (int[]){2, 2, 2, 2}, 4, SZ_I));
        MT_ASSERT_TEST("4D split[1] data",
                       mt_arr_same(result[1]->data,
                                   (mt_float[]){2, 3, 5, 6, 8, 9, 11, 12, 14,
                                                15, 17, 18, 20, 21, 23, 24},
                                   16, SZ_F));

        mt_tensor_free(t);
        mt_tensor_free(result[0]);
        mt_tensor_free(result[1]);
    }

    // 5D tensor split test
    {
        MT_SECTION_TITLE("5D tensor split");
        int        shape[] = {2, 2, 2, 2, 3};
        mt_tensor *t       = mt_tensor_alloc_values(
            shape, 5,
            (mt_float[]){1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,

                               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,

                               25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,

                               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48});
        int        splits[] = {1, 1};
        mt_tensor *result[2];
        mt_tensor_split(t, 1, splits, 2, result);

        MT_ASSERT_TEST(
            "5D split[0] shape",
            mt_arr_same(result[0]->shape, (int[]){2, 1, 2, 2, 3}, 5, SZ_I));
        MT_ASSERT_TEST("5D split[0] data",
                       mt_arr_same(result[0]->data,
                                   (mt_float[]){1,  2,  3,  4,  5,  6,
                                                7,  8,  9,  10, 11, 12,

                                                25, 26, 27, 28, 29, 30,
                                                31, 32, 33, 34, 35, 36},
                                   24, SZ_F));
        MT_ASSERT_TEST(
            "5D split[1] shape",
            mt_arr_same(result[1]->shape, (int[]){2, 1, 2, 2, 3}, 5, SZ_I));
        MT_ASSERT_TEST("5D split[1] data",
                       mt_arr_same(result[1]->data,
                                   (mt_float[]){13, 14, 15, 16, 17, 18,
                                                19, 20, 21, 22, 23, 24,

                                                37, 38, 39, 40, 41, 42,
                                                43, 44, 45, 46, 47, 48},
                                   24, SZ_F));

        mt_tensor_free(t);
        mt_tensor_free(result[0]);
        mt_tensor_free(result[1]);
    }
}

void test_conv2d(Stats *stats) {
    // clang-format off
    MT_SECTION_TITLE("Basic 2D convolution (NCHW)");
    {
        // Input: 1x1x3x3 (NCHW)
        mt_tensor *input = mt_tensor_alloc_values(MT_ARR_INT(1, 1, 3, 3), 4,
                                                  MT_ARR_FLOAT(1, 2, 3,
                                                               4, 5, 6,
                                                               7, 8, 9));
        // Weight: 1x1x2x2
        mt_tensor *weight = mt_tensor_alloc_values(MT_ARR_INT(1, 1, 2, 2), 4,
                                                   MT_ARR_FLOAT(1, 2,
                                                                3, 4));
        // Bias: 1
        mt_tensor *bias = mt_tensor_alloc_values(MT_ARR_INT(1), 1, MT_ARR_FLOAT(1));
        
        int stride = 1;
        int pads[4] = {0, 0, 0, 0};  // top, left, bottom, right
        
        mt_tensor *output = mt_convolve_2d(input, weight, bias, stride, pads);
        
        MT_ASSERT_TEST("Basic conv2d shape", mt_arr_same(output->shape, MT_ARR_INT(1, 1, 2, 2), 4, SZ_I));
        MT_ASSERT_TEST("Basic conv2d data", mt_arr_same(output->data, MT_ARR_FLOAT(38, 48, 68, 78), 4, SZ_F));
        
        mt_tensor_free(input);
        mt_tensor_free(weight);
        mt_tensor_free(bias);
        mt_tensor_free(output);
    }
    
    MT_SECTION_TITLE("Conv2D with padding (NCHW)");
    {
        // Input: 1x1x3x3 (NCHW)
        mt_tensor *input = mt_tensor_alloc_values(MT_ARR_INT(1, 1, 3, 3), 4,
                                                  MT_ARR_FLOAT(1, 2, 3,
                                                               4, 5, 6,
                                                               7, 8, 9));
        // Weight: 1x1x3x3
        mt_tensor *weight = mt_tensor_alloc_values(MT_ARR_INT(1, 1, 3, 3), 4,
                                                   MT_ARR_FLOAT(1, 2, 3,
                                                                4, 5, 6,
                                                                7, 8, 9));
        // Bias: 1
        mt_tensor *bias = mt_tensor_alloc_values(MT_ARR_INT(1), 1, MT_ARR_FLOAT(1));
        
        int stride = 1;
        int pads[4] = {1, 1, 1, 1};  // top, left, bottom, right
        
        mt_tensor *output = mt_convolve_2d(input, weight, bias, stride, pads);
        
        MT_ASSERT_TEST("Padded conv2d shape", mt_arr_same(output->shape, MT_ARR_INT(1, 1, 3, 3), 4, SZ_I));
        mt_float expected[] = {
            95.0, 155.0, 107.0,
            187.0, 286.0, 187.0,
            107.0, 155.0, 95.0
        };
        MT_ASSERT_TEST("Padded conv2d data", mt_arr_same(output->data, expected, 9, SZ_F));
        
        mt_tensor_free(input);
        mt_tensor_free(weight);
        mt_tensor_free(bias);
        mt_tensor_free(output);
    }
    
    MT_SECTION_TITLE("Conv2D with stride (NCHW)");
    {
        // Input: 1x1x5x5 (NCHW)
        mt_tensor *input = mt_tensor_alloc_values(MT_ARR_INT(1, 1, 5, 5), 4,
                                                  MT_ARR_FLOAT(1, 2, 3, 4, 5,
                                                               6, 7, 8, 9, 10,
                                                               11, 12, 13, 14, 15,
                                                               16, 17, 18, 19, 20,
                                                               21, 22, 23, 24, 25));
        // Weight: 1x1x3x3
        mt_tensor *weight = mt_tensor_alloc_values(MT_ARR_INT(1, 1, 3, 3), 4,
                                                   MT_ARR_FLOAT(1, 2, 3,
                                                                4, 5, 6,
                                                                7, 8, 9));
        // Bias: 1
        mt_tensor *bias = mt_tensor_alloc_values(MT_ARR_INT(1), 1, MT_ARR_FLOAT(1));
        
        int stride = 2;
        int pads[4] = {0, 0, 0, 0};  // top, left, bottom, right
        
        mt_tensor *output = mt_convolve_2d(input, weight, bias, stride, pads);
        
        MT_ASSERT_TEST("Strided conv2d shape", mt_arr_same(output->shape, MT_ARR_INT(1, 1, 2, 2), 4, SZ_I));
        mt_float expected[] = {
            412.0, 502.0,
            862.0, 952.0,
        };
        MT_ASSERT_TEST("Strided conv2d data", mt_arr_same(output->data, expected, 4, SZ_F));
        
        mt_tensor_free(input);
        mt_tensor_free(weight);
        mt_tensor_free(bias);
        mt_tensor_free(output);
    }
    
    MT_SECTION_TITLE("Conv2D with multiple input and output channels (NCHW)");
    {
        // Input: 1x2x3x3 (NCHW)
        mt_tensor *input = mt_tensor_alloc_values(MT_ARR_INT(1, 2, 3, 3), 4,
                                                  MT_ARR_FLOAT(1, 2, 3,
                                                               4, 5, 6,
                                                               7, 8, 9,
                                                               
                                                               10, 11, 12,
                                                               13, 14, 15,
                                                               16, 17, 18));
        // Weight: 2x2x2x2
        mt_tensor *weight = mt_tensor_alloc_values(MT_ARR_INT(2, 2, 2, 2), 4,
                                                   MT_ARR_FLOAT(1, 2,
                                                                3, 4,
                                                                
                                                                5, 6,
                                                                7, 8,
                                                                
                                                                9, 10,
                                                                11, 12,
                                                                
                                                                13, 14,
                                                                15, 16));
        // Bias: 2
        mt_tensor *bias = mt_tensor_alloc_values(MT_ARR_INT(2), 1, MT_ARR_FLOAT(1, 2));
        
        int stride = 1;
        int pads[4] = {0, 0, 0, 0};  // top, left, bottom, right
        
        mt_tensor *output = mt_convolve_2d(input, weight, bias, stride, pads);
        
        MT_ASSERT_TEST("Multi-channel conv2d shape", mt_arr_same(output->shape, MT_ARR_INT(1, 2, 2, 2), 4, SZ_I));
        mt_float expected[] = {
            357.0, 393.0,
            465.0, 501.0,

            838.0, 938.0,
            1138.0, 1238.0
        };
        MT_ASSERT_TEST("Multi-channel conv2d data", mt_arr_same(output->data, expected, 8, SZ_F));
        
        mt_tensor_free(input);
        mt_tensor_free(weight);
        mt_tensor_free(bias);
        mt_tensor_free(output);
    }
    
    MT_SECTION_TITLE("Conv2D with batch size > 1 (NCHW)");
    {
        // Input: 2x1x3x3 (NCHW, batch size 2)
        mt_tensor *input = mt_tensor_alloc_values(MT_ARR_INT(2, 1, 3, 3), 4,
                                                  MT_ARR_FLOAT(1, 2, 3,
                                                               4, 5, 6,
                                                               7, 8, 9,
                                                               
                                                               10, 11, 12,
                                                               13, 14, 15,
                                                               16, 17, 18));
        // Weight: 1x1x2x2
        mt_tensor *weight = mt_tensor_alloc_values(MT_ARR_INT(1, 1, 2, 2), 4,
                                                   MT_ARR_FLOAT(1, 2,
                                                                3, 4));
        // Bias: 1
        mt_tensor *bias = mt_tensor_alloc_values(MT_ARR_INT(1), 1, MT_ARR_FLOAT(1));
        
        int stride = 1;
        int pads[4] = {0, 0, 0, 0};  // top, left, bottom, right
        
        mt_tensor *output = mt_convolve_2d(input, weight, bias, stride, pads);
        
        MT_ASSERT_TEST("Batched conv2d shape", mt_arr_same(output->shape, MT_ARR_INT(2, 1, 2, 2), 4, SZ_I));
        mt_float expected[] = {
            38.0, 48.0,
            68.0, 78.0,

            128.0, 138.0,
            158.0, 168.0
        };
        MT_ASSERT_TEST("Batched conv2d data", mt_arr_same(output->data, expected, 8, SZ_F));
        
        mt_tensor_free(input);
        mt_tensor_free(weight);
        mt_tensor_free(bias);
        mt_tensor_free(output);
    }

    MT_SECTION_TITLE("Conv2D with 5 output channels (NCHW)");
    {
        // Input: 1x3x4x4 (NCHW)
        mt_tensor *input = mt_tensor_alloc_values(MT_ARR_INT(1, 3, 4, 4), 4,
                                                  MT_ARR_FLOAT(
                                                    1,  2,  3,  4,
                                                    5,  6,  7,  8,
                                                    9, 10, 11, 12,
                                                   13, 14, 15, 16,

                                                   17, 18, 19, 20,
                                                   21, 22, 23, 24,
                                                   25, 26, 27, 28,
                                                   29, 30, 31, 32,

                                                   33, 34, 35, 36,
                                                   37, 38, 39, 40,
                                                   41, 42, 43, 44,
                                                   45, 46, 47, 48
                                                  ));
        
        // Weight: 5x3x2x2
        mt_tensor *weight = mt_tensor_alloc_values(MT_ARR_INT(5, 3, 2, 2), 4,
                                                   MT_ARR_FLOAT(
                                                     1,  2,  3,  4,   5,  6,  7,  8,   9, 10, 11, 12,
                                                    13, 14, 15, 16,  17, 18, 19, 20,  21, 22, 23, 24,
                                                    25, 26, 27, 28,  29, 30, 31, 32,  33, 34, 35, 36,
                                                    37, 38, 39, 40,  41, 42, 43, 44,  45, 46, 47, 48,
                                                    49, 50, 51, 52,  53, 54, 55, 56,  57, 58, 59, 60
                                                   ));
        
        // Bias: 5
        mt_tensor *bias = mt_tensor_alloc_values(MT_ARR_INT(5), 1, MT_ARR_FLOAT(1, 2, 3, 4, 5));
        
        int stride = 2;
        int pads[4] = {0, 0, 0, 0};  // top, left, bottom, right
        
        mt_tensor *output = mt_convolve_2d(input, weight, bias, stride, pads);
        
        MT_ASSERT_TEST("5 output channels conv2d shape", mt_arr_same(output->shape, MT_ARR_INT(1, 5, 2, 2), 4, SZ_I));
        
        mt_float expected[] = {
            2061.0, 2217.0,
            2685.0, 2841.0,

            4870.0, 5314.0,
            6646.0, 7090.0,

            7679.0, 8411.0,
            10607.0, 11339.0,

            10488.0, 11508.0,
            14568.0, 15588.0,

            13297.0, 14605.0,
            18529.0, 19837.0
        };
        
        MT_ASSERT_TEST("5 output channels conv2d data", mt_arr_same(output->data, expected, 5 * 2 * 2, SZ_F));
        
        mt_tensor_free(input);
        mt_tensor_free(weight);
        mt_tensor_free(bias);
        mt_tensor_free(output);
    }
}

int main() {
    Stats s = {0};

    test_binop(&s);
    test_permute_dim(&s);
    test_slice(&s);
    test_concat(&s);
    test_tensor_pad(&s);
    test_tensor_split(&s);
    test_conv2d(&s);

    printf("FAILED: %d, PASSED: %d\n", s.failed, s.pass);

    return 0;
}

// clang-format on
