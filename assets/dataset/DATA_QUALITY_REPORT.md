# Data Quality Report
**Dataset:** Pakistan Air Quality & Weather (10 Cities)
**Report generated:** 2026-08-13
**Version:** v2 (corrected)

## Summary
During independent EDA, a user (Zainab) identified that weather variables (temperature, humidity, precipitation, wind_speed, wind_direction, pressure) were flat/repeated across all 24 hours for a 21-day window at the start of the dataset (Nov 6-26, 2025), across all 10 cities. This was verified, root-caused, and fixed in v2. Pollutant columns (PM2.5, PM10, CO, NO2, SO2, O3, Dust) were unaffected throughout.

## Issue Identified
- **Affected window:** 2025-11-06 to 2025-11-26 (21 days, 504 hours per city)
- **Affected rows:** 5,040 out of 21,840 total rows (23.1%)
- **Affected columns:** temperature, humidity, precipitation, wind_speed, wind_direction, pressure
- **Unaffected columns:** all pollutant metrics (pm10, pm2_5, carbon_monoxide, nitrogen_dioxide, sulphur_dioxide, ozone, dust), all temporal/categorical features
- **Root cause:** Original collection pipeline likely used a daily-summary weather call (or a pre-hourly-logging fallback) for the first 21 days before switching to true hourly retrieval on Nov 27, 2025. Air quality data was pulled from a separate endpoint that was hourly-resolution from day one.

## Verification Method
1. Grouped data by `date` and computed `nunique()` per weather column for each city.
2. Confirmed all 10 cities showed exactly 1 unique value per weather column for every day in the Nov 6-26 window (fully flat).
3. Confirmed the same window showed normal variation (11-24 unique values/day) for pollutant columns, isolating the fault to weather only.
4. Confirmed unique-value counts jumped to 11-24 starting Nov 27, matching expected hourly variation for the rest of the dataset.

## Before / After Comparison (Example: Lahore, Nov 6-10)
| date       |   temperature_before |   humidity_before |   precipitation_before |   wind_speed_before |   wind_direction_before |   pressure_before |   temperature_after |   humidity_after |   precipitation_after |   wind_speed_after |   wind_direction_after |   pressure_after |
|:-----------|---------------------:|------------------:|-----------------------:|--------------------:|------------------------:|------------------:|--------------------:|-----------------:|----------------------:|-------------------:|-----------------------:|-----------------:|
| 2025-11-06 |                    1 |                 1 |                      1 |                   1 |                       1 |                 1 |                  22 |               18 |                     1 |                 19 |                     19 |               19 |
| 2025-11-07 |                    1 |                 1 |                      1 |                   1 |                       1 |                 1 |                  24 |               21 |                     1 |                 19 |                     20 |               16 |
| 2025-11-08 |                    1 |                 1 |                      1 |                   1 |                       1 |                 1 |                  22 |               19 |                     1 |                 17 |                     21 |               19 |
| 2025-11-09 |                    1 |                 1 |                      1 |                   1 |                       1 |                 1 |                  22 |               19 |                     1 |                 18 |                     21 |               21 |
| 2025-11-10 |                    1 |                 1 |                      1 |                   1 |                       1 |                 1 |                  24 |               19 |                     1 |                 18 |                     21 |               17 |

## Fix Applied
- Re-fetched true hourly weather for all 10 cities, Nov 6-26, 2025, from Open-Meteo's Historical Weather API (`archive-api.open-meteo.com`), using each city's exact lat/lon already present in the dataset.
- Rows replaced: **5040**
- NaNs introduced by the fix: **0** (0 = clean merge)
- Pollutant and temporal columns were left completely untouched.

## Post-Fix Validation
- Re-ran the uniqueness check on the corrected data: Nov 6-26 now shows 14-24 unique weather values/day per city, consistent with the rest of the dataset.
- Final row count unchanged: **21840** rows.
- Final missing value count: **0**

## Acknowledgment
Issue identified and reported by Zainab (BS CS student), while using this dataset for an independent project on smog prediction in Lahore and Punjab cities. Thank you for the careful EDA and for flagging this constructively.

## Conclusion
With this fix, the dataset (v2) is verified to have consistent, genuine hourly resolution across all 26 columns, all 10 cities, and the full 90-day period (Nov 6, 2025 - Feb 4, 2026), with zero missing values and no flat/placeholder windows remaining.