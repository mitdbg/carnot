WITH elephant_counts AS (SELECT city, count(*) as cnt FROM AudioData WHERE Animal = 'Elephant' GROUP BY city) SELECT city FROM elephant_counts WHERE cnt = (SELECT MAX(cnt) FROM elephant_counts);
