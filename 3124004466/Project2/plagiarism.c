#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
#include <ctype.h>
#include <math.h>
#include <windows.h>
#include "plagiarism.h"

// ==================== 哈希表实现 ====================
typedef struct HashNode {
	unsigned int key;   // 2个wchar_t拼接成的键
	int count;          // 出现次数
	struct HashNode* next;
} HashNode;

struct HashTable {
	HashNode** buckets; // 桶数组
	int bucketNum;      // 桶数量
};

HashTable* hash_create(int bucketNum) {
	HashTable* table = (HashTable*)malloc(sizeof(HashTable));
	if (!table) return NULL;
	table->bucketNum = bucketNum;
	table->buckets = (HashNode**)calloc(bucketNum, sizeof(HashNode*));
	if (!table->buckets) {
		free(table);
		return NULL;
	}
	return table;
}

static unsigned int hash_func(unsigned int key, int bucketNum) {
	return key % bucketNum;
}

void hash_insert(HashTable* table, unsigned int key) {
	if (!table) return;
	unsigned int idx = hash_func(key, table->bucketNum);
	HashNode* node = table->buckets[idx];

	while (node) {
		if (node->key == key) {
			node->count++;
			return;
		}
		node = node->next;
	}

	HashNode* newNode = (HashNode*)malloc(sizeof(HashNode));
	if (!newNode) return;
	newNode->key = key;
	newNode->count = 1;
	newNode->next = table->buckets[idx];
	table->buckets[idx] = newNode;
}

int hash_get(HashTable* table, unsigned int key) {
	if (!table) return 0;
	unsigned int idx = hash_func(key, table->bucketNum);
	HashNode* node = table->buckets[idx];

	while (node) {
		if (node->key == key) return node->count;
		node = node->next;
	}
	return 0;
}

void hash_destroy(HashTable* table) {
	if (!table) return;
	for (int i = 0; i < table->bucketNum; i++) {
		HashNode* node = table->buckets[i];
		while (node) {
			HashNode* tmp = node;
			node = node->next;
			free(tmp);
		}
	}
	free(table->buckets);
	free(table);
}

// ==================== 文本与文件处理 ====================

// 判断是否为需要过滤的字符（标点、空白、换行）
static int is_filter_char(wchar_t c) {
	if (c <= 127) {
		return iswpunct(c) || iswspace(c);
	}
	// 中文常见标点
	wchar_t cn_punct[] = L"，。；：“”‘’（）【】、！？《》「」";
	for (int i = 0; cn_punct[i] != L'\0'; i++) {
		if (c == cn_punct[i]) return 1;
	}
	return 0;
}

// 读取UTF-8文件，返回宽字符串，失败返回NULL
wchar_t* read_utf8_file(const char* filePath) {
	FILE* fp = fopen(filePath, "rb");
	if (!fp) return NULL;

	fseek(fp, 0, SEEK_END);
	long fileSize = ftell(fp);
	rewind(fp);

	if (fileSize == 0) {
		fclose(fp);
		wchar_t* empty = (wchar_t*)malloc(sizeof(wchar_t));
		if (empty) empty[0] = L'\0';
		return empty;
	}

	char* buf = (char*)malloc(fileSize + 1);
	if (!buf) {
		fclose(fp);
		return NULL;
	}
	fread(buf, 1, fileSize, fp);
	buf[fileSize] = '\0';
	fclose(fp);

	// 多字节转宽字符（UTF-8 -> Unicode）
	int wLen = MultiByteToWideChar(CP_UTF8, 0, buf, -1, NULL, 0);
	wchar_t* wstr = (wchar_t*)malloc(wLen * sizeof(wchar_t));
	if (!wstr) {
		free(buf);
		return NULL;
	}
	MultiByteToWideChar(CP_UTF8, 0, buf, -1, wstr, wLen);
	free(buf);
	return wstr;
}

// 文本预处理：过滤所有标点和空白
wchar_t* preprocess(const wchar_t* text) {
	if (!text) return NULL;
	int len = wcslen(text);
	wchar_t* res = (wchar_t*)malloc((len + 1) * sizeof(wchar_t));
	if (!res) return NULL;

	int j = 0;
	for (int i = 0; i < len; i++) {
		if (!is_filter_char(text[i])) {
			res[j++] = text[i];
		}
	}
	res[j] = L'\0';
	return res;
}

// 构建2-gram频率哈希表
HashTable* build_2gram_map(const wchar_t* text) {
	if (!text) return NULL;
	int len = wcslen(text);
	if (len < 2) return hash_create(128); // 文本不足2字符，返回空表

	HashTable* table = hash_create(4096);
	if (!table) return NULL;

	for (int i = 0; i < len - 1; i++) {
		// 两个宽字符拼接为一个unsigned int作为哈希键
		unsigned int key = ((unsigned int)text[i] << 16) | (unsigned short)text[i + 1];
		hash_insert(table, key);
	}
	return table;
}

// ==================== 相似度计算 ====================
float calc_cosine_similarity(HashTable* mapA, HashTable* mapB) {
	if (!mapA || !mapB) return 0.0f;

	double dotProduct = 0.0;
	double normA = 0.0;
	double normB = 0.0;

	// 遍历A计算点积和A的模长
	for (int i = 0; i < mapA->bucketNum; i++) {
		HashNode* node = mapA->buckets[i];
		while (node) {
			int cntA = node->count;
			normA += (double)cntA * cntA;
			int cntB = hash_get(mapB, node->key);
			if (cntB > 0) {
				dotProduct += (double)cntA * cntB;
			}
			node = node->next;
		}
	}

	// 计算B的模长
	for (int i = 0; i < mapB->bucketNum; i++) {
		HashNode* node = mapB->buckets[i];
		while (node) {
			normB += (double)node->count * node->count;
			node = node->next;
		}
	}

	if (normA == 0 || normB == 0) return 0.0f;
	return (float)(dotProduct / (sqrt(normA) * sqrt(normB)));
}

// 写入结果到文件，保留两位小数
int write_result(const char* filePath, float rate) {
	FILE* fp = fopen(filePath, "w");
	if (!fp) return -1;
	fprintf(fp, "%.2f", rate);
	fclose(fp);
	return 0;
}
