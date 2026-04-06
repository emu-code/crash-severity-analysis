## Key Insights Summary

## NUMERICAL FEATURES
1. Severity (Target Variable):
   - Imbalanced distribution with majority of Level 2 and 3 accidents
   - This imbalance needs to be considered during model training
   
2. Temperature & Humidity:
   - Show relatively normal distributions
   - Moderate to high variance indicating diverse weather conditions
   
3. Visibility & Precipitation:
   - Highly right-skewed: most accidents occur in normal visibility/no precipitation
   - Severe weather conditions are rarer but may have higher accident severity
   
4. Wind Speed:
   - Most days have calm to moderate wind
   - Few instances of high wind speeds

5. Distance:
   - Highly variable road extent affected by accidents
   - Indicates range from localized to widespread impact

## CATEGORICAL FEATURES
1. Geographic Distribution:
   - Accidents concentrated in specific states
   - State and City features will be important for modeling
   
2. Weather Conditions:
   - Clear weather dominates the dataset
   - Sunny, Cloudy, and Rainy conditions are most common
   
3. Time of Day:
   - Day/Night distribution via Sunrise_Sunset, twilight features
   - May be non-uniform distribution across times
   
4. Wind Direction:
   - Variable and cardinal directions present
   - Few instances of 'CALM' designation

## BOOLEAN/POI FEATURES
1. Traffic Signal Presence:
   - Most prevalent POI feature (present in many accident locations)
   - Indicates urban/developed areas
   
2. Junction Presence:
   - Very common in accident locations
   - Key infrastructure feature affecting accident severity
   
3. Amenity Features:
   - Stop signs, Bumps, and Crossings show varying prevalence
   - Less common features may have higher predictive value if correlated with severity
   
4. Low Prevalence Features:
   - Railway, Station, Roundabout are less common
   - May have specialized predictive value

