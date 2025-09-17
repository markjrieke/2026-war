# CHANGELOG

## war 1.2.1

* Add new arguments to `WARModel().prep_stan_data()` that allow for using a subset of default variables to fit the model.

## war 1.2.0

* Add support for modeling senate results.
* Rearrange contents of `out/` to support both house and senate results.

## war 1.1.0

* Split the "generic challenger" candidate by partisan affiliation.
* Included candidates who have run multiple times as named candidates with unique skill estimates.
* Switched to modeling the outcome in terms of the democrat's share of FEC contribution, rather than logit(dem share of FEC contributions).

## war 1.0.0

* Initial release of house estimates!