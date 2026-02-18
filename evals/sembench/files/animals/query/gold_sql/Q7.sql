select city from (select city from ImageData where Species LIKE '%ZEBRA%') INTERSECT (select city from ImageData where Species LIKE '%IMPALA%');
