select distinct city from ImageData I where Species LIKE '%MONKEY%' and not exists (select * from AudioData A where A.city = I.city and A.animal = 'Monkey');
