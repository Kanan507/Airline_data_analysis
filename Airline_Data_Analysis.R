library(tidymodels)
library(modeltime)
library(tidyverse)
library(lubridate)
library(timetk)
library(earth)

# This toggles plots from plotly (interactive) to ggplot (static)
interactive <- FALSE

# Step 1 - Collect data and split into training and test sets.----
# Data
df <- fread('AirPassengers (3).csv')
names(df)[2] <- 'Count'

df %>% glimpse()

df$Month <- paste(df$Month,"-01",sep = "") 
df$Month <- as.Date(df$Month)


#visualize
df %>% plot_time_series(Month,Count,.interactive = interactive)

# Split Data 80/20
splits <- initial_time_split(df, prop = 0.8)


#Step 2 - Create & Fit Multiple Models----

# Model 2: arima_boost ----
model_fit_arima_boosted <- arima_boost(
  min_n = 2,
  learn_rate = 0.015
) %>%
  set_engine(engine = "auto_arima_xgboost") %>%
  fit(Count ~ Month + as.numeric(Month) + factor(month(Month, label = TRUE), ordered = F),
      data = training(splits))
#> frequency = 12 observations per 1 year

# Model 3: ets ----
model_fit_ets <- exp_smoothing() %>%
  set_engine(engine = "ets") %>%
  fit(Count ~ Month, data = training(splits))
#> frequency = 12 observations per 1 year

# Model 4: prophet ----
model_fit_prophet <- prophet_reg() %>%
  set_engine(engine = "prophet") %>%
  fit(Count ~ Month, data = training(splits))
#> Disabling weekly seasonality. Run prophet with weekly.seasonality=TRUE to override this.
#> Disabling daily seasonality. Run prophet with daily.seasonality=TRUE to override this.




#Step 3 - Add fitted models to a Model Table.----

models_tbl <- modeltime_table(
  model_fit_arima_boosted,
  model_fit_ets,
  model_fit_prophet
)

models_tbl
#> # Modeltime Table
#> # A tibble: 6 x 3
#>   .model_id .model     .model_desc                              
#>       <int> <list>     <chr>                                    
#> 1         1 <fit[+]>   ARIMA(0,1,1)(0,1,1)[12]                  
#> 2         2 <fit[+]>   ARIMA(0,1,1)(0,1,1)[12] W/ XGBOOST ERRORS
#> 3         3 <fit[+]>   ETS(M,A,A)                               
#> 4         4 <fit[+]>   PROPHET                                  
#> 5         5 <fit[+]>   LM                                       
#> 6         6 <workflow> EARTH



#Step 4 - Calibrate the model to a testing set.----
calibration_tbl <- models_tbl %>%
  modeltime_calibrate(new_data = testing(splits))

calibration_tbl
#> # Modeltime Table
#> # A tibble: 6 x 5
#>   .model_id .model     .model_desc                        .type .calibration_da…
#>       <int> <list>     <chr>                              <chr> <list>          
#> 1         1 <fit[+]>   ARIMA(0,1,1)(0,1,1)[12]            Test  <tibble [31 × 4…
#> 2         2 <fit[+]>   ARIMA(0,1,1)(0,1,1)[12] W/ XGBOOS… Test  <tibble [31 × 4…
#> 3         3 <fit[+]>   ETS(M,A,A)                         Test  <tibble [31 × 4…
#> 4         4 <fit[+]>   PROPHET                            Test  <tibble [31 × 4…
#> 5         5 <fit[+]>   LM                                 Test  <tibble [31 × 4…
#> 6         6 <workflow> EARTH                              Test  <tibble [31 × 4…


#Step 5 - Testing Set Forecast & Accuracy Evaluation----
#5A - Visualizing the Forecast Test----
calibration_tbl %>%
  modeltime_forecast(
    new_data    = testing(splits),
    actual_data = df
  ) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = interactive
  )


#5B - Accuracy Metrics----
calibration_tbl %>%
  modeltime_accuracy() %>%
  table_modeltime_accuracy(
    .interactive = interactive
  )

#Step 6 - Refit to Full Dataset & Forecast Forward----

refit_tbl <- calibration_tbl %>%
  modeltime_refit(data = df)

refit_tbl %>%
  modeltime_forecast(h = "3 years", actual_data = df) %>%
  plot_modeltime_forecast(
    .legend_max_width = 25, # For mobile screens
    .interactive      = interactive
  )
