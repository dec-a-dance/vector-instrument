## Vector instrument

Данный проект содержит прототип инструмента для реализации векторного поиска в SQLite базах данных.
Прототип реализован с акцентом на последующее использование в игровых движках, поэтому является утилитой, вызываемой через системное API (консольные команды).
Любой движок может вызывать системные утилиты через subprocess.   
Команды, возможные для вызова:  
Считывание из файлов разных форматов:  
```
vector-cli ingest --file ./data.csv

vector-cli ingest --file ./data.json

vector-cli ingest --file ./input.db --table vectors_in
```
Получение субъективного вектора и его обновление:  
```
vector-cli set-subjective --vector "[0.1, 0.2, 0.3]"` 

vector-cli get-subjective 
```
Поддержка kNN с возможной дальнейшей фильтрацией:  
```
vector-cli knn --vector "[0.2, 0.3, 0.4]" --k 3

vector-cli knn-filtered --vector "[0.2, 0.3, 0.4]" --k 3 --weight 0.7 --threshold 0.5 
``` 
Сброс данных:  
```
vector-cli reset
```
Формат ответов:
```
{"ok": true, "result": {...}}  

{"ok": false, "error": "message"}
```
