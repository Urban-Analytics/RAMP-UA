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


test_that("covid_prob", {

  df <- list(id = sample(1:100, 10, replace = FALSE),
              current_risk = runif(10, 0, 1),
              beta0 = rep(0, 10),
              betaxs = rep(0, 10),
              hid_status = rep(0, 10),
              presymp_days = sample(1:10, size = 10, replace = TRUE),
              symp_days = sample(5:15, size = 10, replace = TRUE),
              status = c(rep(0,5), 1, 2, 3, 4, 0),
              new_status = c(rep(0,5), 1, 2, 3, 4, 0))

  expect_type(covid_prob(df = df, betas = list(current_risk = 0.42), risk_cap_val = 5), "list")

  expect_equal(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["id"]], df$id)
  expect_equal(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["current_risk"]], df$current_risk)
  expect_equal(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["beta0"]], df$beta0)
  expect_equal(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["hid_status"]], df$hid_status)
  expect_equal(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["presymp_days"]], df$presymp_days)
  expect_equal(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["symp_days"]], df$symp_days)
  expect_equal(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["status"]], df$status)
  expect_equal(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["new_status"]], df$new_status)

  expect_type(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["betaxs"]], "double")
  expect_type(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["probability"]], "double")

  expect_length(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["betaxs"]], length(df$betaxs))
  expect_length(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["probability"]], length(df$betaxs))

  expect_lte(max(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["probability"]]), 1)
  expect_gte(min(covid_prob(df = df,betas = list(current_risk = 0.42), risk_cap_val = 5)[["probability"]]), 0)
})

test_that("case_assign works", {

  df <- list(id = sample(1:100, 10, replace = FALSE),
             current_risk = runif(10, 0, 1),
             beta0 = rep(0, 10),
             betaxs = runif(10, 0, 1),
             hid_status = rep(0, 10),
             presymp_days = sample(1:10, size = 10, replace = TRUE),
             symp_days = sample(5:15, size = 10, replace = TRUE),
             status = c(rep(0,5), 1, 2, 3, 4, 0),
             new_status = c(rep(0,5), 1, 2, 3, 4, 0),
             probability = runif(10, 0, 1))

  expect_true(all(case_assign(df)[["new_status"]] >= df$new_status))
  expect_true(all(case_assign(df)[["new_status"]] >= df$status))

  expect_type(case_assign(df), "list")

  expect_equal(case_assign(df)[["id"]], df$id)
  expect_equal(case_assign(df)[["current_risk"]], df$current_risk)
  expect_equal(case_assign(df)[["beta0"]], df$beta0)
  expect_equal(case_assign(df)[["betaxs"]], df$betaxs)
  expect_equal(case_assign(df)[["hid_status"]], df$hid_status)
  expect_equal(case_assign(df)[["presymp_days"]], df$presymp_days)
  expect_equal(case_assign(df)[["symp_days"]], df$symp_days)
  expect_equal(case_assign(df)[["status"]], df$status)
  expect_equal(case_assign(df)[["probability"]], df$probability)
})


test_that("rank_assign works", {
  df <- list(id = sample(1:100, 10, replace = FALSE),
             current_risk = runif(10, 0, 1),
             beta0 = rep(0, 10),
             betaxs = runif(10, 0, 1),
             hid_status = rep(0, 10),
             presymp_days = sample(1:10, size = 10, replace = TRUE),
             symp_days = sample(5:15, size = 10, replace = TRUE),
             status = c(rep(0,5), 1, 2, 3, 4, 0),
             new_status = c(rep(0,5), 1, 2, 3, 4, 0),
             probability = runif(10, 0, 1))

  daily_case <- 10

  expect_true(all(rank_assign(df, daily_case = daily_case)[["new_status"]] >= df$new_status))
  expect_true(all(rank_assign(df, daily_case = daily_case)[["new_status"]] >= df$status))

  expect_type(rank_assign(df, daily_case = daily_case), "list")

  expect_equal(rank_assign(df, daily_case = daily_case)[["id"]], df$id)
  expect_equal(rank_assign(df, daily_case = daily_case)[["current_risk"]], df$current_risk)
  expect_equal(rank_assign(df, daily_case = daily_case)[["beta0"]], df$beta0)
  expect_equal(rank_assign(df, daily_case = daily_case)[["betaxs"]], df$betaxs)
  expect_equal(rank_assign(df, daily_case = daily_case)[["hid_status"]], df$hid_status)
  expect_equal(rank_assign(df, daily_case = daily_case)[["presymp_days"]], df$presymp_days)
  expect_equal(rank_assign(df, daily_case = daily_case)[["symp_days"]], df$symp_days)
  expect_equal(rank_assign(df, daily_case = daily_case)[["status"]], df$status)
  expect_equal(rank_assign(df, daily_case = daily_case)[["probability"]], df$probability)
})


test_that("infection_length works", {

  df <- list(id = 1:10,
             current_risk = runif(10, 0, 1),
             beta0 = rep(0, 10),
             betaxs = runif(10, 0, 1),
             hid_status = rep(0, 10),
             presymp_days = rep(0,10),
             symp_days = rep(0,10),
             status = c(rep(0,5), 1, 2, 3, 4, 0),
             new_status = c(rep(1,5), 2, 3, 4, 4, 0),
             probability = runif(10, 0, 1))

  timestep <- 1

  expect_type(infection_length(df, timestep = timestep), "list")

  expect_true(all(infection_length(df, timestep = timestep)[["new_status"]] >= df$new_status))
  expect_true(all(infection_length(df, timestep = timestep)[["new_status"]] >= df$status))

  expect_equal(infection_length(df, timestep = timestep)[["id"]], df$id)
  expect_equal(infection_length(df, timestep = timestep)[["current_risk"]], df$current_risk)
  expect_equal(infection_length(df, timestep = timestep)[["beta0"]], df$beta0)
  expect_equal(infection_length(df, timestep = timestep)[["betaxs"]], df$betaxs)
  expect_equal(infection_length(df, timestep = timestep)[["hid_status"]], df$hid_status)
  expect_equal(infection_length(df, timestep = timestep)[["status"]], df$status)
  expect_equal(infection_length(df, timestep = timestep)[["probability"]], df$probability)
  expect_equal(sum(infection_length(df, timestep = timestep)[["presymp_days"]] >0) - sum(df$presymp_days > 0),sum(df$status == 0 & df$new_status == 1))

})

test_that("removed works", {

  df <- list(id = 1:10,
             current_risk = runif(10, 0, 1),
             beta0 = rep(0, 10),
             betaxs = runif(10, 0, 1),
             hid_status = rep(0, 10),
             presymp_days = rep(0,10),
             symp_days = rep(1,10),
             status = c(rep(0,5), rep(2, 5)),
             new_status =c(rep(0,5), rep(2, 5)),
             probability = runif(10, 0, 1))
  timestep <- 1
  expect_type(infection_length(df, timestep = timestep), "list")

  expect_true(all(removed(df)[["new_status"]] >= df$new_status))
  expect_true(all(removed(df)[["new_status"]] >= df$status))



})

test_that("normalizer works", {
  expect_equal(normalizer(0.5, 0,1,0.5,1), 0)
  expect_equal(normalizer(0.75, 0,1,0.5,1), 0.5)
  expect_equal(normalizer(1, 0,1,0.5,1), 1)
  expect_equal(normalizer(0, 0, 1, 0.5, 1), -1)
})
