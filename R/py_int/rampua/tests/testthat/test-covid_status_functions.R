test_that("multiplication works", {
  expect_equal(2 * 2, 4)
})

df <- data.frame(presymp_days = sample(1:10, size = 10, replace = TRUE),
                 symp_days = sample(5:15, size = 10, replace = TRUE),
                 disease_status = c(rep(0,5), 1, 2, 3, 4, 0),
                 current_risk  = runif(10, 0, 1))

test_that("create_input works",{
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

dfl <- create_input(df, "current_risk")

test_that("covid_prob", {

})


test_that("normalizer works", {
  expect_equal(normalizer(0.5, 0,1,0.5,1), 0)
  expect_equal(normalizer(0.75, 0,1,0.5,1), 0.5)
  expect_equal(normalizer(1, 0,1,0.5,1), 1)
  expect_equal(normalizer(0, 0, 1, 0.5, 1), -1)
})
