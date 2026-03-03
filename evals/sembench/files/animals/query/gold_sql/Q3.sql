WITH zebra_counts AS (SELECT city, count(*) as cnt FROM ImageData WHERE Species LIKE '%ZEBRA%' GROUP BY city) SELECT city FROM zebra_counts WHERE cnt = (SELECT MAX(cnt) FROM zebra_counts);
