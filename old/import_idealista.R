library(dplyr) # A Grammar of Data Manipulation
library(ggplot2) # Create Elegant Data Visualisations Using the Grammar of Graphics
library(idealista18) # Idealista 2018 Data Package
library(sf) # Simple Features for R
library(repr) 
library(skimr) 

Barcelona_Sale |>
  sf::st_drop_geometry() |> 
  skimr::skim()


st_write(Barcelona_Sale, "C:/Users/1738037/OneDrive - UAB/1- CLIMGROW Charlotte/1- PSC cities/data_barcelona/barcelona_rent_idealista.gpkg", layer = "points", driver = "GPKG")