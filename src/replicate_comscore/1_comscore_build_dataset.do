
local input_dir = "../../data/comscore/input"
local output_dir = "../../data/comscore/intermediate"

use "`input_dir'/selected_categories/master_list_categories.dta", clear
levelsof cat_code4, local(cat_values)

foreach category in `cat_values' {

	*** Import list of ASINS for estimation ***
	import delimited "`input_dir'/selected_products/`category'/asin_list.csv", varnames(1) clear stringcols(2)
	rename asin asin_code
	keep asin_code
	sort asin_code
	gen product_id = _n
	scalar num_items = _N
	save "`output_dir'/`category'/asins_selected.dta", replace
	export delimited using "`output_dir'/`category'/asin_to_item_crosswalk.csv", replace

	*** Build list of all purchases in category ***
	use "`input_dir'/purchases/amazon_crosswalk.dta", clear
	rename product_asin asin_code
	merge m:1 asin_code using "`input_dir'/categories/all_variations2.dta", keep(1 3) nogenerate
	replace asin_pooled = asin_code if asin_pooled == ""
	drop asin_code
	rename asin_pooled asin_code
	merge m:1 asin_code using "`output_dir'/`category'/asins_selected.dta", keep(match) nogenerate
	gen date = date(substr(event_time, 1, 10), "YMD")
	format %d date
	drop if machine_id == .
	keep machine_id date product_id
	sort machine_id date
	gen order_id = _n
	save "`output_dir'/`category'/choices.dta", replace

	*** Build purchases and prices ***
	* Build a list of purchases *
	use "`output_dir'/`category'/choices.dta", clear
	keep order_id date product_id
	expand num_items
	bysort order_id: gen num = _n
	gen purchase = (product_id == num)
	drop product_id
	rename num product_id
	sort order_id product_id
	keep order_id product_id date purchase
	merge m:1 product_id using "`output_dir'/`category'/asins_selected.dta", keep(match) nogenerate
	* Merge purchases with prices *
	merge m:1 asin_code date using "`input_dir'/prices/prices.dta", keep(1 3) nogenerate
	merge m:1 asin_code using "`input_dir'/prices/prices_extra.dta", keep(1 3 4) update nogenerate // manually collected missing prices
	gsort asin_code date
	by asin_code: replace price = price[_n-1] if price[_n-1]~=. & price == .
	gsort asin_code -date
	by asin_code: replace price = price[_n-1] if price[_n-1]~=. & price == .
	merge m:1 asin_code using "`input_dir'/prices/prices_avg.dta", update
	drop if _merge == 2
	drop asin_code _merge
	drop date
	sort order_id
	* Assert that each order_id records a purchase
	bysort order_id: egen any_purchase = max(purchase)
	assert any_purchase == 1
	drop any_purchase
	order order_id product_id purchase price
	* Save data for this category
	outsheet using "`output_dir'/`category'/long_format_data.csv", comma replace
	reshape wide purchase price, i(order_id) j(product_id)
	outsheet order_id purchase*   using "`output_dir'/`category'/matrix_purchases.csv", comma replace
	outsheet order_id price*      using "`output_dir'/`category'/matrix_prices.csv",    comma replace
	
	*** Remove temporary files ***
	erase "`output_dir'/`category'/choices.dta"
	erase "`output_dir'/`category'/asins_selected.dta"

}

