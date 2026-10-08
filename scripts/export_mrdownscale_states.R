# Export mrdownscale's downscaled land use as a CSV that graft can score.
#
# calcLandHighRes returns a magpie object: area in Mha per 0.25 degree cell,
# with crops split by irrigation and biofuel type (c3ann_irrigated,
# c4per_rainfed_biofuel_2nd_gen, ...). This writes x, y, year and one column
# per LUH state, each crop subtype summed onto its LUH state, still in Mha.
# graft.io.mrdownscale.read_states_csv turns a year of it into LUH state
# fractions on the full grid.
#
# The one step in the chain that needs R, because the object is R's. Needs
# magclass.
#
#   Rscript scripts/export_mrdownscale_states.R highres.rds states.csv.gz [2025 2050 2100]

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  stop("usage: Rscript export_mrdownscale_states.R <highres.rds> <out.csv(.gz)> [years ...]")
}
suppressMessages(library(magclass))

x <- readRDS(args[1])
years <- if (length(args) > 2) as.integer(args[-(1:2)]) else getYears(x, as.integer = TRUE)
missingYears <- setdiff(years, getYears(x, as.integer = TRUE))
if (length(missingYears) > 0) stop("years not in ", args[1], ": ", paste(missingYears, collapse = ", "))

categories <- getItems(x, dim = 3)
# a crop subtype's LUH state is the part before the first underscore
state <- ifelse(grepl("^c[34]", categories), sub("_.*$", "", categories), categories)
coords <- getCoords(x)

out <- do.call(rbind, lapply(years, function(year) {
  rows <- data.frame(x = coords$x, y = coords$y, year = year)
  for (s in unique(state)) {
    rows[[s]] <- as.vector(dimSums(x[, year, categories[state == s]], dim = 3))
  }
  rows
}))

connection <- if (grepl("[.]gz$", args[2])) gzfile(args[2]) else args[2]
utils::write.csv(out, connection, row.names = FALSE)
cat("wrote", nrow(out), "rows:", length(years), "years x", nrow(coords), "cells,",
    length(unique(state)), "states\n")
