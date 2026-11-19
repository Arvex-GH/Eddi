"""Money and economics: the facts a person actually needs about money.

Everyday finance is the subject people ask about most and know least, and it
is almost entirely absent from the other packs. The facts here are deliberately
concrete: what a term means, how an instrument works, and what the trade-off
is. Prose carries the reasoning; facts carry the definitions so a direct
question can be answered without a retrieval guess.
"""

from __future__ import annotations

from typing import List, Tuple

Fact = Tuple[str, str, str]

FACTS: List[Fact] = [
    # ------------------------------------------------------------ interest
    ("interest", "definition", "price_paid_for_borrowing_money"),
    ("compound_interest", "definition", "interest_that_also_earns_interest"),
    ("compound_interest", "property", "grows_exponentially_with_time"),
    ("compound_interest", "effect", "time_matters_more_than_the_rate"),
    ("simple_interest", "definition", "interest_paid_only_on_the_original_amount"),
    ("principal", "definition", "original_amount_borrowed_or_invested"),
    ("annual_percentage_rate", "definition", "yearly_cost_of_credit_including_fees"),
    ("rule_of_72", "definition", "divide_72_by_the_rate_to_approximate_years_to_double"),
    # ---------------------------------------------------------- inflation
    ("inflation", "definition", "general_rise_in_the_price_level"),
    ("inflation", "effect", "money_loses_purchasing_power"),
    ("inflation", "measured_by", "consumer_price_index"),
    ("deflation", "definition", "general_fall_in_the_price_level"),
    ("real_return", "definition", "return_after_subtracting_inflation"),
    ("nominal_return", "definition", "return_before_subtracting_inflation"),
    # ----------------------------------------------------------- investing
    ("share", "definition", "unit_of_ownership_in_a_company"),
    ("bond", "definition", "loan_to_a_government_or_company"),
    ("dividend", "definition", "part_of_profit_paid_to_shareholders"),
    ("stock_market", "function", "lets_people_buy_and_sell_shares"),
    ("index_fund", "definition", "fund_tracking_a_broad_market_index"),
    ("index_fund", "advantage", "low_costs_through_passive_holding"),
    ("diversification", "purpose", "reduce_risk_by_spreading_holdings"),
    ("risk", "definition", "chance_that_a_return_differs_from_expectation"),
    ("volatility", "definition", "how_much_a_price_swings_over_time"),
    ("asset", "definition", "something_of_value_that_can_be_owned"),
    ("liability", "definition", "something_owed_to_someone_else"),
    ("net_worth", "definition", "assets_minus_liabilities"),
    ("liquidity", "definition", "how_easily_an_asset_can_be_sold_without_loss"),
    # ----------------------------------------------------------- borrowing
    ("credit_score", "definition", "number_summarising_creditworthiness"),
    ("mortgage", "definition", "loan_secured_against_a_property"),
    ("collateral", "definition", "asset_pledged_to_secure_a_loan"),
    ("default", "definition", "failure_to_repay_a_debt_on_time"),
    ("bankruptcy", "definition", "legal_process_for_debts_that_cannot_be_paid"),
    ("usury", "definition", "charging_interest_at_an_excessive_rate"),
    # ------------------------------------------------------------ banking
    ("bank", "function", "takes_deposits_and_makes_loans"),
    ("bank_run", "definition", "many_depositors_withdrawing_at_once"),
    ("deposit_insurance", "purpose", "prevent_bank_runs"),
    ("central_bank", "function", "sets_interest_rates_and_controls_money_supply"),
    ("commercial_bank", "function", "holds_deposits_and_lends_to_people"),
    ("fractional_reserve_banking", "property", "banks_lend_out_most_of_what_they_hold"),
    ("money_supply", "definition", "total_amount_of_money_in_an_economy"),
    # ---------------------------------------------------------------- tax
    ("tax", "definition", "compulsory_payment_to_a_government"),
    ("progressive_tax", "definition", "higher_rates_on_higher_incomes"),
    ("regressive_tax", "definition", "takes_a_larger_share_from_lower_incomes"),
    ("vat", "definition", "tax_on_value_added_at_each_stage_of_production"),
    ("income_tax", "definition", "tax_on_money_earned"),
    ("capital_gains_tax", "definition", "tax_on_profit_from_selling_an_asset"),
    # --------------------------------------------------------- insurance
    ("insurance", "function", "pools_risk_across_many_people"),
    ("premium", "definition", "regular_payment_for_insurance_cover"),
    ("deductible", "definition", "amount_paid_by_the_insured_before_cover_starts"),
    ("moral_hazard", "definition", "taking_more_risk_because_someone_else_bears_the_cost"),
    ("adverse_selection", "definition", "the_wrong_people_choosing_to_buy_insurance"),
    # ------------------------------------------------------------ economy
    ("gross_domestic_product", "definition", "total_value_of_goods_and_services_produced"),
    ("recession", "definition", "two_consecutive_quarters_of_falling_output"),
    ("unemployment", "definition", "share_of_people_wanting_work_and_not_finding_it"),
    ("productivity", "definition", "output_per_hour_worked"),
    ("budget_deficit", "definition", "government_spending_exceeding_revenue_in_a_year"),
    ("national_debt", "definition", "total_money_a_government_has_borrowed"),
    ("trade_deficit", "definition", "imports_exceeding_exports"),
    ("tariff", "definition", "tax_on_imported_goods"),
    ("subsidy", "definition", "government_payment_that_lowers_a_cost"),
    ("monopoly", "definition", "single_seller_controlling_a_market"),
    ("supply_and_demand", "function", "sets_price_where_willingness_meets_availability"),
    ("market_clearing_price", "definition", "price_at_which_supply_equals_demand"),
    ("elasticity", "definition", "how_much_quantity_responds_to_a_price_change"),
    ("opportunity_cost", "definition", "value_of_the_best_alternative_given_up"),
    ("sunk_cost", "property", "already_spent_and_should_not_affect_future_decisions"),
    ("invisible_hand", "definition", "prices_coordinating_self_interested_choices"),
]

PROSE: List[str] = [
    "Compound interest earns interest on the interest already earned, so the balance grows faster the longer it is left.",
    "The rule of 72 estimates doubling time by dividing 72 by the annual percentage return.",
    "A dollar invested at seven percent doubles in roughly ten years, which is why starting early beats saving more later.",
    "Inflation means money kept in a drawer loses value, so a return below the inflation rate is still a loss.",
    "A real return subtracts inflation from the nominal return, which is the only figure that measures purchasing power.",
    "Diversification lowers risk without necessarily lowering expected return, because different assets do not fall at the same moment.",
    "An index fund buys the whole market cheaply, which beats most active funds over long periods because costs compound against you.",
    "Fees compound exactly like returns, but against the investor rather than for them.",
    "Volatility is not the same as risk: a temporary fall only becomes a loss when it forces a sale.",
    "A bond pays a fixed interest payment, so its price falls when interest rates rise and new bonds pay more.",
    "A share is a claim on a company's future profit, which is why its price moves with expectations rather than with current earnings alone.",
    "Debt secured against an asset costs less than unsecured debt because the lender can recover value if the borrower defaults.",
    "A mortgage is the largest loan most people ever take, and the interest rate fixes how much the same house costs over thirty years.",
    "A credit score summarises how reliably a person has repaid past debt, so it prices future borrowing.",
    "Paying a high interest debt is the same as earning that rate risk free, which is why it beats most investments.",
    "An emergency fund is not an investment: it exists so that a shock does not force a sale at the worst moment.",
    "Insurance is worth buying for losses that would be catastrophic and not for losses that would be merely annoying.",
    "A deductible lowers the premium by moving small claims onto the insured, which is a sensible trade for most people.",
    "Moral hazard appears when somebody else bears the consequences of your risk taking.",
    "Adverse selection means the people most likely to claim are also the most likely to buy, which is why insurance is often mandatory.",
    "A bank lends out most of what it holds, which is why a rumour can destroy a solvent bank.",
    "Deposit insurance exists to stop runs before they start, because a run is caused by the expectation of a run.",
    "A central bank raises interest rates to slow borrowing and spending, which is the main tool against inflation.",
    "Inflation is measured on a basket of goods, so an individual's experience depends on what they actually buy.",
    "A progressive tax charges a higher rate on higher income, and a regressive one takes a larger share from the poor.",
    "A tariff protects domestic producers and raises prices for domestic consumers.",
    "A subsidy lowers a price the government wants lower, and someone always pays for it through tax.",
    "A monopoly charges more and produces less than a competitive market would.",
    "The market clearing price is where the quantity people want to buy equals the quantity others want to sell.",
    "Elasticity measures how much a quantity responds to a price change, and it decides who really pays a tax.",
    "Opportunity cost is what you gave up, which is why a free option is rarely free.",
    "A sunk cost is already spent, so it should never decide what happens next.",
    "Prices carry information that no single person holds, which is what an invisible hand amounts to in practice.",
    "A trade deficit is not a debt: it means a country bought more than it sold, and the difference was invested or borrowed.",
    "Gross domestic product counts output, not wellbeing, so it misses unpaid care and environmental damage.",
    "Productivity growth is the only lasting source of rising wages.",
    "A recession is usually defined as two quarters of falling output, though the dating is done by judgement.",
    "National debt is owed mostly to a country's own citizens and institutions through bonds and pensions.",
    "A deficit is not automatically bad: borrowing to build infrastructure can pay for itself.",
]
