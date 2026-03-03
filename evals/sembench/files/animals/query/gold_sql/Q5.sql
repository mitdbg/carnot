select distinct city from (select city from ImageData where Species LIKE '%ELEPHANT%') UNION (select city from AudioData where Animal = 'Elephant');
