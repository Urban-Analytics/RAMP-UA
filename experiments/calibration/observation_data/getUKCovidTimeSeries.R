library(tidyverse)

getUKCovidTimeseries <- function (){
  UKregional = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTzgWD-_-vxj7ljb-iYPckgtV4ctg-SjVJWQzrjwj0CWF2JE9uyLSUwtQFZal3Cdqf-5Mch-_sBPBv2/pub?gid=163112336&single=true&output=csv", 
                               col_types = readr::cols(date = readr::col_date(format = "%Y-%m-%d")))
  UKregional$uk_cumulative_cases[72:nrow(UKregional)] <- UKregional$england_cumulative_cases[72:nrow(UKregional)] + UKregional$scotland_cumulative_cases[72:nrow(UKregional)] + UKregional$wales_cumulative_cases[72:nrow(UKregional)] + UKregional$northern_ireland_cumulative_cases[72:nrow(UKregional)]
  englandNHS = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTzgWD-_-vxj7ljb-iYPckgtV4ctg-SjVJWQzrjwj0CWF2JE9uyLSUwtQFZal3Cdqf-5Mch-_sBPBv2/pub?gid=0&single=true&output=csv", 
                               col_types = readr::cols(date = readr::col_date(format = "%Y-%m-%d")))
  scotlandHealthBoard = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQod-HdDk4Nl8BFcunG5P-QA2CuKdIXCfK53HJDxcsaYlOov4FFc-yQciJyQFrqX5_n_ixz56S7uNBh/pub?gid=490497042&single=true&output=csv")
  walesHealthBoard = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQod-HdDk4Nl8BFcunG5P-QA2CuKdIXCfK53HJDxcsaYlOov4FFc-yQciJyQFrqX5_n_ixz56S7uNBh/pub?gid=762770891&single=true&output=csv")
  northernIreland = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQod-HdDk4Nl8BFcunG5P-QA2CuKdIXCfK53HJDxcsaYlOov4FFc-yQciJyQFrqX5_n_ixz56S7uNBh/pub?gid=1217212942&single=true&output=csv")
  englandUnitAuth = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vTzgWD-_-vxj7ljb-iYPckgtV4ctg-SjVJWQzrjwj0CWF2JE9uyLSUwtQFZal3Cdqf-5Mch-_sBPBv2/pub?gid=796246456&single=true&output=csv")
  englandUnitAuth2NHSregion = readr::read_csv("https://docs.google.com/spreadsheets/d/e/2PACX-1vQod-HdDk4Nl8BFcunG5P-QA2CuKdIXCfK53HJDxcsaYlOov4FFc-yQciJyQFrqX5_n_ixz56S7uNBh/pub?gid=1933702254&single=true&output=csv")
  tmp = englandUnitAuth %>% tidyr::pivot_longer(cols = starts_with("20"), 
                                                names_to = "date", values_to = "cumulative_cases") %>% 
    mutate(date = as.Date(as.character(date), "%Y-%m-%d"))
  tmp = tmp %>% left_join(UKregional %>% select(date, daily_total = england_cumulative_cases), 
                          by = "date")
  tidyEnglandUnitAuth = tmp %>% group_by(date) %>% mutate(daily_unknown = daily_total - 
                                                            sum(cumulative_cases, na.rm = TRUE)) %>% ungroup() %>% 
    group_by(CTYUA19CD, CTYUA19NM)
  tmp = englandNHS %>% tidyr::pivot_longer(cols = !date, names_to = "england_nhs_region", 
                                           values_to = "cumulative_cases")
  tmp = tmp %>% left_join(UKregional %>% select(date, daily_total = england_cumulative_cases), 
                          by = "date")
  tidyEnglandNHS = tmp %>% group_by(date) %>% mutate(daily_unknown = daily_total - 
                                                       sum(cumulative_cases, na.rm = TRUE)) %>% ungroup() %>% 
    group_by(england_nhs_region)
  tidyUKRegional = UKregional %>% select(date, england_cumulative_cases, 
                                         scotland_cumulative_cases, wales_cumulative_cases, northern_ireland_cumulative_cases, 
                                         daily_total = uk_cumulative_cases) %>% tidyr::pivot_longer(cols = ends_with("cumulative_cases"), 
                                                                                                    names_to = "uk_region", values_to = "cumulative_cases") %>% 
    filter(!is.na(cumulative_cases)) %>% mutate(uk_region = stringr::str_remove(uk_region, 
                                                                                "_cumulative_cases")) %>% mutate(uk_region = stringr::str_replace(uk_region, 
                                                                                                                                                  "_", " ")) %>% group_by(date) %>% mutate(daily_unknown = daily_total - 
                                                                                                                                                                                             sum(cumulative_cases, na.rm = TRUE)) %>% ungroup() %>% 
    group_by(uk_region)
  return(list(UKregional = UKregional, englandNHS = englandNHS, 
              englandUnitAuth = englandUnitAuth, scotlandHealthBoard = scotlandHealthBoard, 
              walesHealthBoard = walesHealthBoard, northernIrelandLocalGovernmentDistrict = northernIreland, 
              englandUnitAuth2NHSregion = englandUnitAuth2NHSregion, 
              tidyUKRegional = tidyUKRegional, tidyEnglandNHS = tidyEnglandNHS, 
              tidyEnglandUnitAuth = tidyEnglandUnitAuth))
}
