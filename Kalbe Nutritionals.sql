
-- Average customer age based on their marital status

select "Marital Status", avg(age) as Age_Average
from customer c 
group by "Marital Status" 

-- Average customer age based on their gender

select gender, avg(age) as Age_Average
from customer c 
group by gender -- 0 = female, 1 = male

-- Store with the highest total quantity
select storename, sum(qty) as Total_Quantity
from "Transaction" t 
join store s on t.storeid = s.storeid 
group by storename
order by Total_Quantity desc
limit 1

-- Best-selling product with the highest total amount
select "Product Name", sum(totalamount) as Sum_Amount
from product p 
join "Transaction" t on t.productid = p.productid 
group by "Product Name" 
order by Sum_Amount desc
limit 1


