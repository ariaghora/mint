#define MT_IMPLEMENTATION
#include "../mint.h"

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

#define MT_ARR_SAME(arr1, arr2, len)                                           \
    ({                                                                         \
        int are_same = 1;                                                      \
        for (int i = 0; i < len; i++) {                                        \
            if ((arr1)[i] != (arr2)[i]) {                                      \
                are_same = 0;                                                  \
                break;                                                         \
            }                                                                  \
        }                                                                      \
        are_same;                                                              \
    })

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
                   MT_ARR_SAME(res_add->data, MT_ARR_FLOAT(3, 6), 2));
    MT_ASSERT_TEST("simple sub",
                   MT_ARR_SAME(res_sub->data, MT_ARR_FLOAT(-1, -2), 2));
    MT_ASSERT_TEST("simple mul",
                   MT_ARR_SAME(res_mul->data, MT_ARR_FLOAT(2, 8), 2));
    MT_ASSERT_TEST("simple div",
                   MT_ARR_SAME(res_div->data, MT_ARR_FLOAT(0.5, 0.5), 2));

    MT_TENSOR_BULK_FREE(6, a, b, res_add, res_sub, res_mul, res_div);

    MT_SECTION_TITLE("broadcast binop (2, 2) (2)");
    a = mt_tensor_alloc_values(MT_ARR_INT(2, 2), 2, MT_ARR_FLOAT(1, 2, 3, 4));
    b = mt_tensor_alloc_values(MT_ARR_INT(2), 1, MT_ARR_FLOAT(1, 2));
    res_add = mt_add(a, b);
    MT_ASSERT_TEST("add shape",
                   MT_ARR_SAME(res_add->shape, MT_ARR_INT(2, 2), 2));
    MT_ASSERT_TEST("add data",
                   MT_ARR_SAME(res_add->data, MT_ARR_FLOAT(2, 4, 4, 6), 4));
}

int main() {
    Stats s = {0};

    test_binop(&s);

    printf("FAILED: %d, PASSED: %d\n", s.failed, s.pass);
    return 0;
}