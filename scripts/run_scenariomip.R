#!/usr/bin/env Rscript
# Write a ScenarioMIP product from a staged IAMC source.
#
# The one bridge into the fork for a full run: scripts/prepare_iamc_source.py
# stages ~/madrat/sources/IAMC for one marker, and this turns it into the
# three netCDF files plus consistencyCheck.log. fullSCENARIOMIP writes into
# the working directory, so each run gets its own folder under the madrat
# output directory.
#
#   Rscript scripts/run_scenariomip.R "High - SSP3 (Marker)" h-iamc [MRDOWNSCALE]
#
# Arguments: the scenario as the staged data.csv names it, a tag for the
# filenames and the folder, and optionally the path to the mrdownscale clone.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: run_scenariomip.R <scenario> <tag> [gross,fold,harvest] [mrdownscale path]")
scenario <- args[1]
tag <- args[2]
opts <- if (length(args) >= 3) strsplit(tolower(args[3]), ",")[[1]] else character(0)
gross <- any(opts %in% c("gross", "true", "1"))
fold <- any(opts %in% c("fold", "foldplantations"))
# primary forest driven by the scenario's wood demand (fork branch
# graft/primary-forest-harvest) instead of fadeForest's historical extrapolation
harmonization <- if (any(opts %in% c("harvest", "fadeforestharvest"))) "fadeForestHarvest" else "fadeForest"
pkg <- if (length(args) >= 4) args[4] else "~/GitHub/mrdownscale"
# last year to write; 2150 for an input extended by scripts/extend_iamc.py
yearEnd <- as.integer(Sys.getenv("YEAR_END", "2100"))

suppressMessages({
  library(madrat)
  pkgload::load_all(path.expand(pkg), quiet = TRUE)
})

# madrat falls back to a temporary main folder when nothing sets one, and
# then finds neither the staged source nor the LUH3 download
mainfolder <- Sys.getenv("MADRAT_MAINFOLDER", path.expand("~/madrat"))
setConfig(mainfolder = mainfolder, .verbose = FALSE)

# Markers can only run side by side if each has its own source folder, because
# staging one overwrites the IAMC source of the last. The cache is deliberately
# shared: its keys carry a fingerprint of what was read, so the LUH3-derived
# steps are reused across markers while the marker-specific ones stay
# distinct. Warm it with one marker serially before fanning out, or several
# processes will compute the same LUH3 target at once and race to write it.
sourcefolder <- Sys.getenv("MADRAT_SOURCEFOLDER", "")
if (nzchar(sourcefolder)) setConfig(sourcefolder = sourcefolder, .verbose = FALSE)
cachefolder <- Sys.getenv("MADRAT_CACHEFOLDER", "")
if (nzchar(cachefolder)) setConfig(cachefolder = cachefolder, .verbose = FALSE)
message("madrat main folder: ", getConfig("mainfolder"))
message("  sources: ", getConfig("sourcefolder"))
message("  cache:   ", getConfig("cachefolder"))

staged <- read.csv(file.path(getConfig("sourcefolder"), "IAMC", "data.csv"), nrows = 1)
message("staged source: ", staged$Model, " / ", staged$Scenario)
if (!identical(trimws(staged$Scenario), trimws(scenario))) {
  stop("the staged source holds ", staged$Scenario, ", not ", scenario,
       " - run scripts/prepare_iamc_source.py for this marker first")
}

# calcOutput(file = ) writes into the output folder while fullSCENARIOMIP's
# metadata step reopens the file by a path relative to the working directory,
# so the two have to be the same folder
out <- file.path(getConfig("outputfolder"), gsub("-", "_", tag))
dir.create(out, recursive = TRUE, showWarnings = FALSE)
setConfig(outputfolder = out, .verbose = FALSE)
old <- setwd(out)
on.exit(setwd(old))

message("writing to ", out)
if (gross) message("writing gross land transitions as well")
if (fold) message("folding plantations into secondary forest, share kept as manaf")
message("harmonization: ", harmonization)
message("years: 2020 to ", yearEnd)
fullSCENARIOMIP(input = paste0("iamc:", scenario), scenario = tag,
                yearsSubset = 2020:yearEnd, grossTransitions = gross,
                foldPlantations = fold, harmonization = harmonization)
message("done: ", paste(list.files(out), collapse = ", "))
