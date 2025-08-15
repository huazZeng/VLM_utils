# Paser
数据解析类，对于llm推理的结果进行解析，如果没有选择parser的情况下，默认返回推理结果
## 如何使用自定义paser
* 在paser文件夹中新建 xxxxPaser类，其中xxxx为类名
* 实现 parser_to_print , parser_to_save 方法
    * print格式 为单次测试时输出的结果
    * save 格式为保存结果
* 在使用unified_inference时，传入参数
```bash
   --parser xxxx
```
