library(tidyverse)

top = read_csv('current_topline.csv')

top = top %>%
  mutate(state = state.abb[match(state_name, state.name)]) %>%
  mutate(WARP = WARP * 100) %>%
  mutate_at(c('WAR','WAR_upper','WAR_lower'),
            function(x){x*200}) %>%
  relocate(state)

write_csv(top, 'current_topline_x2.csv')