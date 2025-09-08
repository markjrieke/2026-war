library(tidyverse)
library(tidycensus)

# Median income ----------------------------------------------------------------

read_income <- function(state_abb) {
  
  read_csv(glue::glue("https://fred.stlouisfed.org/graph/fredgraph.csv?id=MEHOINUS{state_abb}A672N")) %>%
    rename(income = 2) %>%
    mutate(state = state_abb)
  
}

income <- map_dfr(state.abb, read_income)

# census data: age, hispanic, black, asian, urban ------------------------------
# note: tidycensus only has access going back to 2000, need to supplement manually

variables_2020 <-
  tribble(
    ~variable, ~variable_name,
    "P11_002N", "hispanic",
    "P5_004N", "black",
    "P5_006N", "asian",
    "P13_001N", "age",
    "P2_001N", "total",
    "P2_002N", "urban"
  )

variables_2010 <-
  tribble(
    ~variable, ~variable_name,
    "P004003", "hispanic",
    "P003003", "black",
    "P003005", "asian",
    "P003006", "pacific islander",
    "P013001", "age",
    "P002001", "total",
    "P002002", "urban"
  )

variables_2000 <-
  tribble(
    ~variable, ~variable_name,
    "P002001", "total",
    "P002002", "urban",
    "P004002", "hispanic",
    "P008004", "black",
    "P008006", "asian",
    "P013001", "age"
  )

# fetch & wrangle census data
fetch_census <- function(variables, year) {
  
  sumfile <- ifelse(year == 2020, "dhc", "sf1")
  
  census <-
    get_decennial(
      geography = "state",
      variables = variables$variable,
      year = year,
      sumfile = sumfile
    )
  
  census <- 
    census %>%
    left_join(variables) %>%
    select(state = NAME,
           variable = variable_name,
           value) %>%
    mutate(variable = if_else(variable %in% c("asian", "pacific islander"),
                              "asian",
                              variable)) %>%
    group_by(state, variable) %>%
    summarise(value = sum(value)) %>%
    ungroup() %>%
    arrange(variable, state)
  
  census <- 
    census %>%
    filter(variable == "total") %>%
    select(state,
           total = value) %>%
    right_join(census %>% filter(variable != "total")) %>%
    mutate(value = if_else(variable != "age", value/total, value)) %>%
    select(state, variable, value) %>%
    mutate(year = year) %>%
    relocate(year)
  
  return(census)
  
}

quick_pivot <- function(data, year) {
  
  data %>%
    pivot_longer(-state,
                 names_to = "variable",
                 values_to = "value") %>%
    mutate(year = year) %>%
    relocate(year) %>%
    bind_rows(variables)
  
}

# fetch relevant datasets from the 2000/2010/2020 census
variables <- 
  fetch_census(variables_2020, 2020) %>%
  bind_rows(fetch_census(variables_2010, 2010)) %>%
  bind_rows(fetch_census(variables_2000, 2000))

# Add in 1990 demographic information
# Ref: https://www.iowadatacenter.org/datatables/UnitedStates/usstracehispanic1990.pdf
variables <- 
  read_csv("data/state_race_origin_1990_census.csv") %>%
  mutate(across(c(hispanic, black, asian), ~.x/total)) %>%
  select(-total) %>%
  quick_pivot(1990)

# Add in 1990 age information
# Ref: https://www.iowadatacenter.org/datatables/UnitedStates/usstagesel1990.pdf
variables <- 
  read_csv("data/state_age_1990_census.csv") %>%
  quick_pivot(1990)

# Add in 1990 urbanicity
# Ref: https://www.iowadatacenter.org/datatables/UnitedStates/urusstpop19002000.pdf
variables <- 
  read_csv("data/state_urbanicity_1990_census.csv") %>%
  mutate(urban = urban/total) %>%
  select(-total) %>%
  quick_pivot(1990)

# Educational attainment -------------------------------------------------------

# 1990 educational attainment
# Ref: https://www.iowadatacenter.org/datatables/UnitedStates/ussteducation1990.pdf
variables <- 
  read_csv("data/state_educational_attainment_1990_census.csv") %>%
  mutate(colplus = colplus/100) %>%
  quick_pivot(1990)
  
# 2000 educational attainment
# Ref: https://www.iowadatacenter.org/datatables/UnitedStates/ussteducation2000.pdf
variables <- 
  read_csv("data/state_educational_attainment_2000_census.csv") %>%
  mutate(colplus = colplus/100) %>%
  quick_pivot(2000)

# 2010 ACS Estimates
variables_acs <-
  tribble(
    ~variable, ~variable_name,
    "B27019_002", "25-64 total",
    "B27019_023", "65+ total",
    "B27019_018", "25-64 colplus",
    "B27019_039", "64+ colplus"
  )

fetch_acs <- function(variables, year) {
  
  acs <-
    get_acs(
      geography = "state",
      variables = variables$variable,
      year = year,
      survey = "acs1"
    )
  
  out <- 
    acs %>%
    left_join(variables) %>%
    select(state = NAME,
           variable = variable_name,
           value = estimate) %>%
    mutate(variable = if_else(str_detect(variable, "total"), "total", "colplus")) %>%
    group_by(state, variable) %>%
    summarise(value = sum(value)) %>%
    ungroup() %>%
    pivot_wider(names_from = variable,
                values_from = value) %>%
    mutate(colplus = colplus/total) %>%
    select(-total) %>%
    quick_pivot(year)
  
  return(out)
  
}

# 2020 results not published to the census bureau
variables <- fetch_acs(variables_acs, 2010)
variables <- fetch_acs(variables_acs, 2022)

# wrangle and write ------------------------------------------------------------

tibble(state_abb = state.abb,
       state_name = state.name) %>%
  right_join(income, by = c("state_abb" = "state")) %>%
  mutate(year = year(observation_date)) %>%
  select(year,
         state = state_name,
         income) %>%
  pivot_longer(income,
               names_to = "variable",
               values_to = "value") %>%
  bind_rows(variables) %>%
  filter(!str_detect(state, "District|Puerto")) %>%
  write_csv("data/state_demos.csv")
