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

int main() {
    Stats s = {0};

    test_binop(&s);

    printf("FAILED: %d, PASSED: %d\n", s.failed, s.pass);
    return 0;
}
