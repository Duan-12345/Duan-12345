#include <stdio.h>
#include <stdlib.h>
#include <wchar.h>
#include <assert.h>
#include <locale.h>
#include "plagiarism.h"
#include <string.h>
// 测试1：空文本预处理
void test_preprocess_empty() {
	wchar_t* res = preprocess(L"");
	assert(wcscmp(res, L"") == 0);
	free(res);
	printf("测试1通过：空文本预处理\n");
}

// 测试2：纯标点文本过滤
void test_preprocess_punct() {
	wchar_t* res = preprocess(L"，。！？abc123  ");
	assert(wcscmp(res, L"abc123") == 0);
	free(res);
	printf("测试2通过：纯标点文本过滤\n");
}

// 测试3：正常中文预处理
void test_preprocess_normal() {
	wchar_t* res = preprocess(L"今天，天气晴。");
	assert(wcscmp(res, L"今天天气晴") == 0);
	free(res);
	printf("测试3通过：正常中文文本过滤\n");
}

// 测试4：哈希表插入与查询
void test_hash_insert_get() {
	HashTable* table = hash_create(128);
	hash_insert(table, 0x12345678);
	hash_insert(table, 0x12345678);
	assert(hash_get(table, 0x12345678) == 2);
	assert(hash_get(table, 0x87654321) == 0);
	hash_destroy(table);
	printf("测试4通过：哈希表插入查询\n");
}

// 测试5：完全相同文本相似度≈1
void test_same_text() {
	wchar_t* text = L"今天天气好";
	HashTable* a = build_2gram_map(text);
	HashTable* b = build_2gram_map(text);
	float sim = calc_cosine_similarity(a, b);
	assert(sim > 0.99f);
	hash_destroy(a);
	hash_destroy(b);
	printf("测试5通过：相同文本相似度\n");
}

// 测试6：题目示例文本
void test_example_text() {
	wchar_t* orig = L"今天是星期天天气晴今天晚上我要去看电影";
	wchar_t* copy = L"今天是周天天气晴朗我晚上要去看电影";
	HashTable* a = build_2gram_map(orig);
	HashTable* b = build_2gram_map(copy);
	float sim = calc_cosine_similarity(a, b);
	assert(sim > 0.6f && sim < 1.0f);
	hash_destroy(a);
	hash_destroy(b);
	printf("测试6通过：示例文本相似度\n");
}

// 测试7：完全不同文本
void test_different_text() {
	wchar_t* a = L"机器学习人工智能";
	wchar_t* b = L"今天晚上吃什么";
	HashTable* ma = build_2gram_map(a);
	HashTable* mb = build_2gram_map(b);
	float sim = calc_cosine_similarity(ma, mb);
	assert(sim < 0.2f);
	hash_destroy(ma);
	hash_destroy(mb);
	printf("测试7通过：不同文本相似度\n");
}

// 测试8：单字符文本（长度不足2）
void test_single_char() {
	wchar_t* a = L"好";
	wchar_t* b = L"好";
	HashTable* ma = build_2gram_map(a);
	HashTable* mb = build_2gram_map(b);
	float sim = calc_cosine_similarity(ma, mb);
	assert(sim == 0.0f);
	hash_destroy(ma);
	hash_destroy(mb);
	printf("测试8通过：单字符文本处理\n");
}

// 测试9：空文本相似度为0
void test_empty_similarity() {
	HashTable* ma = build_2gram_map(L"");
	HashTable* mb = build_2gram_map(L"测试");
	float sim = calc_cosine_similarity(ma, mb);
	assert(sim == 0.0f);
	hash_destroy(ma);
	hash_destroy(mb);
	printf("测试9通过：空文本相似度\n");
}

// 测试10：结果写入格式校验
void test_write_result() {
	const char* path = "test_out.txt";
	write_result(path, 0.856f);
	FILE* fp = fopen(path, "r");
	char buf[10] = { 0 };
	fgets(buf, 10, fp);
	fclose(fp);
	assert(strcmp(buf, "0.86") == 0);
	remove(path);
	printf("测试10通过：结果写入格式\n");
}

// 测试11：长文本相似度
void test_long_text() {
	wchar_t* orig = L"计算机技术推动社会进步人工智能在各领域广泛应用深度学习在图像识别自然语言处理中取得突破";
	wchar_t* copy = L"计算机科技促进社会进步AI在很多领域广泛使用深度学习在图像识别NLP中取得突破性成果";
	HashTable* ma = build_2gram_map(orig);
	HashTable* mb = build_2gram_map(copy);
	float sim = calc_cosine_similarity(ma, mb);
	assert(sim > 0.5f);
	hash_destroy(ma);
	hash_destroy(mb);
	printf("测试11通过：长文本相似度\n");
}

// 测试12：标点不影响相似度结果
void test_punct_no_effect() {
	wchar_t* a = L"今天天气好";
	wchar_t* b = L"今天，天气。好！";
	wchar_t* cleanB = preprocess(b);
	HashTable* ma = build_2gram_map(a);
	HashTable* mb = build_2gram_map(cleanB);
	float sim = calc_cosine_similarity(ma, mb);
	assert(sim > 0.99f);
	free(cleanB);
	hash_destroy(ma);
	hash_destroy(mb);
	printf("测试12通过：标点不影响相似度\n");
}

int main() {
	setlocale(LC_ALL, "chs");
	
	setvbuf(stdout, NULL, _IONBF, 0);

	test_preprocess_empty();
	test_preprocess_punct();
	test_preprocess_normal();
	test_hash_insert_get();
	test_same_text();
	test_example_text();
	test_different_text();
	test_single_char();
	test_empty_similarity();
	test_write_result();
	test_long_text();
	test_punct_no_effect();

	printf("\n所有测试通过！共12个测试用例\n");
	return 0;
}
