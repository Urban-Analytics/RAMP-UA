test_that("multiplication works", {
  expect_equal(2 * 2, 4)
})



test_that("create_input works",{

  df <- data.frame(presymp_days = sample(1:10, size = 10, replace = TRUE),
                   symp_days = sample(5:15, size = 10, replace = TRUE),
                   disease_status = c(rep(0,5), 1, 2, 3, 4, 0),
                   current_risk  = runif(10, 0, 1))

  expect_type(create_input(micro_sim_pop = df,vars = "current_risk"), "list")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["current_risk"]], "double")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["beta0"]], "double")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["betaxs"]], "double")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["hid_status"]], "double")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["presymp_days"]], "integer")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["symp_days"]], "integer")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["probability"]], "double")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["status"]], "integer")
  expect_type(create_input(micro_sim_pop = df,vars = "current_risk")[["new_status"]], "integer")
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["current_risk"]], nrow(df))
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["beta0"]], nrow(df))
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["betaxs"]], nrow(df))
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["hid_status"]], nrow(df))
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["presymp_days"]], nrow(df))
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["symp_days"]], nrow(df))
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["probability"]], nrow(df))
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["status"]], nrow(df))
  expect_length(create_input(micro_sim_pop = df,vars = "current_risk")[["new_status"]], nrow(df))

})


covid_prob(df = dfl, betas = list(current_risk = 0.42), risk_cap_val = 5)

test_that("covid_prob", {

  dfl <- list(current_risk = runif(10, 0, 1),
              beta0 = rep(0, 10),
              betaxs = rep(0, 10),
              hid_status = rep(0, 10),
              presymp_days = sample(1:10, size = 10, replace = TRUE),
              symp_days = sample(5:15, size = 10, replace = TRUE),
              status = c(rep(0,5), 1, 2, 3, 4, 0),
              new_status = c(rep(0,5), 1, 2, 3, 4, 0))

  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5), "list")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["current_risk"]], "double")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["beta0"]], "double")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["betaxs"]], "double")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["hid_status"]], "double")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["presymp_days"]], "integer")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["symp_days"]], "integer")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["probability"]], "double")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["status"]], "integer")
  expect_type(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["new_status"]], "integer")
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["current_risk"]], nrow(df))
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["beta0"]], nrow(df))
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["betaxs"]], nrow(df))
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["hid_status"]], nrow(df))
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["presymp_days"]], nrow(df))
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["symp_days"]], nrow(df))
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["probability"]], nrow(df))
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["status"]], nrow(df))
  expect_length(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["new_status"]], nrow(df))
  expect_lt(max(covid_prob(df = dfl,betas = list(current_risk = 0.42), risk_cap_val = 5)[["current_risk"]]), 5)
})


test_that("normalizer works", {
  expect_equal(normalizer(0.5, 0,1,0.5,1), 0)
  expect_equal(normalizer(0.75, 0,1,0.5,1), 0.5)
  expect_equal(normalizer(1, 0,1,0.5,1), 1)
  expect_equal(normalizer(0, 0, 1, 0.5, 1), -1)
})
