library(tidyverse)
library(arrow)
library(janitor)

# read in all pols
pols = read_parquet('full_data.parquet') %>%
  select(M, cycle, state_name, district, dem_pct=pct, candidate_DEM, candidate_REP) %>%
  gather('party','representative', 6:7) %>%
  mutate(party = tolower(gsub('candidate_','',party))) 

# mutate uncontested var
pols = pols %>%
  group_by(M) %>%
  mutate(uncontested = ifelse(any(representative == 'Uncontested'), 1, 0)) %>% 
  ungroup() %>%
  filter(representative != 'Uncontested')

# filter to winning candidates OR candidates who appear twice+
pols = pols %>% 
  group_by(representative) %>%
  mutate(winner = 
           (party == 'dem' & dem_pct > 0.5) | (party == 'rep' & dem_pct < 0.5) ) %>%
  filter( winner |  (n() >= 0) ) %>%
  ungroup() # %>% group_by(representative) %>% filter(n() > 5)

# join WAR
WAR = read_parquet('WAR.parquet') %>%
  filter(quantile == 0.5)

pols = pols %>% 
  left_join(WAR, by = c('M','party')) %>%
  mutate(chamber = 'House') %>%
  select(-quantile)

pols

# by candidate name, join with something we can join to voteview
cand = map_df(list.files('cand/',full.names = T),
       function(fn){
         read_csv(fn) %>% 
           mutate(seat = as.numeric(seat))
         }
       ) %>%
  mutate(chamber = ifelse(is.na(seat),'Senate','House'))

cand = cand %>% 
  mutate(state_name = state.name[match(state,state.abb)] ) %>%
  select(cycle, state_name, chamber, district=seat, representative = candidate,
         fec_id)

pols = pols %>%
  left_join(cand,
            by = c('cycle', 'state_name', 'chamber',
                   'district', 'representative'))

# DIME -- has fec ids
dime = read_csv('dime_recipients_1979_2024.csv')
dime = janitor::clean_names(dime)

dime = dime %>%
  filter(seat %in% c('federal:house','federal:senate')) %>%
  mutate(chamber = case_when(seat == 'federal:house' ~ 'House',
                             seat == 'federal:senate' ~ 'Senate',
                             T ~ NA_character_))


dime = dime %>%
  mutate(state_name = state.name[match(state,state.abb)] ) %>%
  select(chamber, cycle, 
         state_name,
         fec_id = cand_id, 
         score = composite_score,
         gen_vote_pct)

dat = pols %>%
  left_join(dime, by = c('cycle','state_name','chamber','fec_id'))


# relative ideology -- moderation compared to party in year x
dat = dat %>%
  group_by(party) %>%
  mutate(rel_score = (score - mean(score, na.rm = T)),
         rel_score = ifelse(party == 'dem', rel_score * -1, rel_score))


dat %>% 
  group_by(party, cycle) %>%
  mutate(nt = as.character(ntile(rel_score, 5))) %>%
  group_by(party, cycle, nt) %>%
  summarise(mu = mean(score, na.rm = T)) %>%
  ggplot(., aes(x = cycle, y = mu, col = nt)) + 
  geom_line() + 
  facet_wrap(~party) + theme_minimal()


dat %>% 
  group_by(party, cycle) %>%
  summarise(mu = mean(score, na.rm = T)) %>%
  ggplot(., aes(x = cycle, y = mu, col = party)) + 
  geom_line() + theme_minimal()

dat %>% 
  ggplot(., aes(x = score, col = party)) + 
  geom_density() + 
  theme_minimal() + 
  facet_wrap(~cycle) +
  scale_color_manual(values=c('rep'='red','dem'='blue')) +
  coord_cartesian(xlim = c(-5, 5))


dat %>% 
  filter(party == 'dem') %>%
  ggplot(., aes(x = score, col = as.character(cycle))) + 
  geom_density() + 
  theme_minimal() + 
  coord_cartesian(xlim = c(-5, 5))

library(lme4)
model = dat %>%
  filter(uncontested == 0) %>%
  lmer(WAR*100 ~ 1 + abs(score) + (1 + abs(score) | cycle) , data = .)

predict(model,
        newdata = tibble(score = c(-3, -2.6, -1),
                         cycle = 2000))

predict(model,
        newdata = tibble(score = c(-3, -2.6, -1),
                         cycle = 2024))

# -1 jared golden
# -2.6 lloyd doggett/don beyer
# -3 ilhan omar

# eda
ggplot(dat[dat$cycle == 2024,], aes(x = abs(score), y = WAR*100)) + 
  geom_point(shape = 1,aes(col = party),show.legend = F) + 
  scale_color_manual(values=c('dem'='blue','rep'='red')) +
  geom_smooth(method = 'lm',col='gray40') +
  theme_minimal() +
  labs(x = 'Ideological extremity (Bonica "composite score")\n(Lower is more moderate)',
       y = 'Wins Above Replacement (percentage points)',
       subtitle = 'Politician WAR versus ideological extremity, 2024.\nThe most moderate members outperform the most ideologically extreme by ~3 points.',
       title = 'Moderates no longer have significantly higher WAR',
       col = '')

ggplot(dat, aes(x = abs(score), y = WAR*100)) + 
  geom_point(shape = 1) +
  geom_smooth(method = 'lm',aes(col = as.character(cycle)),show.legend = T, se = F) +
  theme_minimal() +
  labs(x = 'Ideological extremity (Bonica "composite score")\n(Lower is more moderate)',
       y = 'Wins Above Replacement (percentage points)',
       subtitle = 'Politician WAR versus ideological extremity, 2000-2024',
       title = 'The electoral advantage of ideological moderation is declining',
       col = 'year')

ggplot(dat, aes(x = abs(score), y = WAR*100)) + 
  geom_point(shape = 1,aes(col = party),show.legend = F) + 
  scale_color_manual(values=c('dem'='blue','rep'='red')) +
  geom_smooth(method = 'lm', aes (col = party), se=F,show.legend = F) +
  facet_wrap(~cycle) + 
  theme_minimal() +
  labs(x = 'Ideological extremity (Bonica "composite score")\n(Lower is more moderate)',
       y = 'Wins Above Replacement (percentage points)',
       subtitle = 'Politician WAR versus ideological extremity, 2000-2024',
       title = 'The electoral advantage of ideological moderation is declining')

lm(gen_vote_pct ~ I(WAR*100),
   data = 
     dat[(dat$uncontested == 0 & 
            dat$gen_vote_pct > 10 & 
            dat$gen_vote_pct < 90 &
            dat$WAR*100 < 10 &
            dat$WAR*100 > -10),]
   ) %>% summary


ggplot(dat[(dat$uncontested == 0 & 
             dat$gen_vote_pct > 10 & 
             dat$gen_vote_pct < 90 &
             dat$WAR*100 < 20 &
             dat$WAR*100 > -20),], 
       aes(x = WAR*100, y = gen_vote_pct-mean(gen_vote_pct,na.rm=T))) + 
  geom_point(shape = 1) + 
  geom_smooth(method = 'lm') +
  geom_abline() +
  theme_minimal() +
  labs(x = 'Wins Above Replacement (percentage points)',
       y = 'Candidate vote share above/below average',
       subtitle = 'Pre-election WAR for candidate versus election result',
       title = 'Candidates with higher WAR do better')


lm(WAR ~ abs(score), 
   data = dat %>% 
     filter(cycle == 2024)) %>% summary

# variance explained by WAR? ----------------------------------------------
# trim data set
temp = dat[(dat$uncontested == 0 & 
               dat$dem_pct > .10 & 
               dat$dem_pct < .90 &
               dat$WAR*100 < 20 &
               dat$WAR*100 > -20 &
              dat$party == 'dem'),]
        
# add race-level vars
races = read_csv('house_forecast_data_updated.csv') %>% select(-uncontested)
dat = dat %>% left_join(races, by = c('cycle','state_name','district'))


# regression on partisanship
lm(dem_pct ~ 1 + dem_pres_twop_lag_lean_one - cycle,
   data = dat[dat$cycle == 2024 & dat$party == 'dem',] ) %>% summary


lm(dem_pct ~ 1  + abs(score) + dem_pres_twop_lag_lean_one - cycle,
   data = dat[dat$cycle == 2024 & dat$party == 'dem',] ) %>% summary


lm(dem_pct ~ 1  + WAR + abs(score) + dem_pres_twop_lag_lean_one - cycle,
   data = dat[dat$cycle == 2024 & dat$party == 'dem',] ) %>% summary


# redo elects, make everyone more moderate -----------------
library(mgcv)
cycle_sims = map_df(seq(2012,2024,2),
       function(yr){
         print(yr)
         # get data
         tmp = dat %>% filter(cycle == yr)
         
         # get dems
         dems = tmp[tmp$party == 'dem',]
         
         # fill in gaps
         ideo_model = lm(score ~ dem_pres_twop_lag_lean, data = dems)
         dems = dems %>%
           mutate(score = ifelse(is.na(score), predict(ideo_model, newdata = .), score))
       
         # bootstrap
         sims = map_df(1:5000,
                function(i){
                  # resample
                  dems = sample_n(tbl = dems, size = nrow(dems), replace = T)
                  
                  # get relationship between covars and vote pct
                  vote_model = lm(dem_pct ~ 1 +
                                    WAR + 
                                    dem_pres_twop_lag_lean + 
                                    incumbent +
                                    experience +
                                    has_fec:dem_share_fec
                                    , data = dems)
                  
                  # get relationship between ideo and WAR, for winners
                  war_model = gam(WAR ~ s(score), data = dems)
                  
                  # now for each candidate, predict WAR based on actual values, and hypo values
                  dems$WAR_hat = predict(war_model, newdata = dems)
                  dems$hypo_WAR_hat = predict(war_model, 
                                              newdata = (dems %>% mutate(score = score + 1)))
                  
                  dems$delta = dems$hypo_WAR_hat - dems$WAR_hat 
                  
                  # add the delta to WAR
                  dems$WAR = dems$WAR + dems$delta
                  
                  # re-predict dem_pct
                  dems$dem_pct_hat = predict(vote_model, newdata = dems)
                  
                  # sum up seats
                  tibble(cycle = yr, iter = i, 
                         dem_chg = sum(dems$dem_pct_hat > 0.5) - 
                           sum(dems$dem_pct > 0.5)) %>%
                    
                    return
                })
         
         return(sims)
         
       })

# summarise
sims = sims %>%
  group_by(cycle) %>%
  summarise(median = median(dem_chg),
            upper = quantile(dem_chg,0.975),
            lower = quantile(dem_chg,0.025)
  )
sims
print(sims)

