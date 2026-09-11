#ifndef PLAGIARISM_H
#define PLAGIARISM_H

#include <wchar.h>

// 哈希表结构体前置声明
typedef struct HashTable HashTable;

// 哈希表操作
HashTable* hash_create(int bucketNum);
void hash_destroy(HashTable* table);
void hash_insert(HashTable* table, unsigned int key);
int hash_get(HashTable* table, unsigned int key);

// 文件与文本处理
wchar_t* read_utf8_file(const char* filePath);
wchar_t* preprocess(const wchar_t* text);
HashTable* build_2gram_map(const wchar_t* text);

// 相似度计算与结果输出
float calc_cosine_similarity(HashTable* mapA, HashTable* mapB);
int write_result(const char* filePath, float rate);

#endif

