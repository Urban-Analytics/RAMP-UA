test_that("load_rpackages works",{
  test_check("rampuaR")
})


test_that("load_init_data works", {

})


test_that("initialize_r works", {

})


test_that("run_status works", {
  df <- data.frame(presymp_days = sample(1:10, size = 10, replace = TRUE),
                   symp_days = sample(5:15, size = 10, replace = TRUE),
                   disease_status = c(rep(0,5), 1, 2, 3, 4, 0),
                   current_risk  = runif(10, 0, 1))

  expect_type(run_status(pop = df)[["betaxs"]], "list")
})