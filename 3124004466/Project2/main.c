#include <stdio.h>
#include <stdlib.h>
#include <locale.h>
#include "plagiarism.h"

int main(int argc, char* argv[]) {
	setlocale(LC_ALL, "chs");

	// 命令行参数校验
	if (argc != 4) {
		printf("参数错误！\n用法：%s 原文路径 抄袭版路径 输出路径\n", argv[0]);
		return 1;
	}

	const char* origPath = argv[1];
	const char* copyPath = argv[2];
	const char* outPath = argv[3];

	// 读取文件
	wchar_t* origText = read_utf8_file(origPath);
	if (!origText) {
		printf("错误：无法读取原文文件 %s\n", origPath);
		return 2;
	}
	wchar_t* copyText = read_utf8_file(copyPath);
	if (!copyText) {
		printf("错误：无法读取抄袭版文件 %s\n", copyPath);
		free(origText);
		return 2;
	}

	// 文本预处理
	wchar_t* cleanOrig = preprocess(origText);
	wchar_t* cleanCopy = preprocess(copyText);
	free(origText);
	free(copyText);

	if (!cleanOrig || !cleanCopy) {
		printf("错误：文本预处理失败\n");
		free(cleanOrig);
		free(cleanCopy);
		return 3;
	}
	
	// 构建2-gram统计模型
	HashTable* origMap = build_2gram_map(cleanOrig);
	HashTable* copyMap = build_2gram_map(cleanCopy);
	free(cleanOrig);
	free(cleanCopy);

	if (!origMap || !copyMap) {
		printf("错误：构建统计模型失败\n");
		hash_destroy(origMap);
		hash_destroy(copyMap);
		return 4;
	}

	// 计算重复率
	float similarity = calc_cosine_similarity(origMap, copyMap);

	// 写入结果
	if (write_result(outPath, similarity) != 0) {
		printf("错误：无法写入输出文件 %s\n", outPath);
		hash_destroy(origMap);
		hash_destroy(copyMap);
		return 5;
	}

	// 释放资源
	hash_destroy(origMap);
	hash_destroy(copyMap);

	printf("计算完成，重复率：%.2f\n", similarity);
	return 0;
}
