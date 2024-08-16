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

    // MT_SECTION_TITLE("1D slice with reverse step");
    // mt_tensor *c = mt_tensor_alloc_values(
    //     MT_ARR_INT(10), 1, MT_ARR_FLOAT(0, 1, 2, 3, 4, 5, 6, 7, 8, 9));
    // int        starts4[] = {9};
    // int        ends4[]   = {-1};
    // int        steps4[]  = {-2};
    // mt_tensor *res4      = mt_tensor_slice(c, starts4, ends4, NULL, steps4,
    // 1); printf("1D slice with reverse step result:\n");
    // MT_ASSERT_TEST("shape", mt_arr_same(res4->shape, MT_ARR_INT(5), 1,
    // SZ_I)); MT_ASSERT_TEST(
    //     "data", mt_arr_same(res4->data, MT_ARR_FLOAT(9, 7, 5, 3, 1), 5,
    //     SZ_F));
    // mt_tensor_free(b);
    // mt_tensor_free(c);
    // mt_tensor_free(res4);

    // MT_SECTION_TITLE("4D slice with partial axes");
    // mt_tensor *d = mt_tensor_alloc_values(
    //     MT_ARR_INT(2, 3, 4, 5), 4, NULL); // Initialize with sequential
    //     values
    // int        starts5[] = {0, 1};
    // int        ends5[]   = {2, 3};
    // int        axes5[]   = {1, 2};
    // mt_tensor *res5      = mt_tensor_slice(d, starts5, ends5, axes5, NULL,
    // 2); MT_ASSERT_TEST("shape",
    //                mt_arr_same(res5->shape, MT_ARR_INT(2, 2, 2, 5), 4,
    //                SZ_I));
    // mt_tensor_free(d);
    // mt_tensor_free(res5);
}

int main() {
    Stats s = {0};

    test_binop(&s);
    test_permute_dim(&s);
    test_slice(&s);

    printf("FAILED: %d, PASSED: %d\n", s.failed, s.pass);
    return 0;
}
