
-- USERS TABLE
WITH users_cte AS (
    SELECT
        *,
        EXTRACT(YEAR FROM AGE(birthdate))::int AS age,
        (CURRENT_DATE - sign_up_date) AS membership_duration_days
    FROM users
),
age_group_cte AS (
    SELECT
        *,
        CASE
          WHEN age < 24 THEN '18-24'
          WHEN age between 25 and 34 THEN '25-34'
          WHEN age between 35 and 44 THEN '35-44'
          WHEN age between 45 and 54 THEN '45-54'
          WHEN age between 55 and 64 THEN '55-64'
          ELSE '65 and older'
        END AS age_group
    FROM users_cte
),
family_status_cte AS (
    SELECT
        *,
        CASE
            WHEN married = FALSE AND has_children = FALSE THEN 'single'
            WHEN married = FALSE AND has_children = TRUE THEN 'single parent'
            WHEN married = TRUE AND has_children = FALSE THEN 'married without children'
            WHEN married = TRUE AND has_children = TRUE THEN 'married with children'
            ELSE 'Unknown'
        END AS family_status
    FROM age_group_cte
),
membership_cte AS (
    SELECT
        *,
        CASE
            WHEN membership_duration_days < 180 THEN 'new'
            WHEN membership_duration_days BETWEEN 180 AND 360 THEN 'short-term'
            WHEN membership_duration_days BETWEEN 360 AND 720 THEN 'mid-term'
            ELSE 'long-term'
        END AS membership_status
    FROM family_status_cte
)
SELECT
    user_id,
    gender,
    age,
    age_group,
    married,
    has_children,
    family_status,
    membership_cte.home_country,
    membership_cte.home_city,
    home_airport,
    sign_up_date,
    membership_duration_days,
    membership_status
FROM membership_cte;



-- HOTELS TABLE
WITH nights_correction AS (
    SELECT
        *,
        ABS(check_out_time::DATE - check_in_time::DATE) AS calc_nights,
        TRIM(TO_CHAR(check_in_time, 'Day')) AS check_in_weekday,
        EXTRACT(HOUR FROM check_in_time) AS check_in_hour
    FROM hotels
),
hotel_costs AS (
    SELECT
        trip_id,
        hotel_name,
        check_in_time,
        check_out_time,
        rooms,
        hotel_per_room_usd,
        calc_nights,
        check_in_weekday,
        check_in_hour,
        (calc_nights * rooms * hotel_per_room_usd) AS total_hotel_amount
    FROM nights_correction
),
percentiles AS (
    SELECT
        percentile_cont(0.10) WITHIN GROUP (ORDER BY hotel_per_room_usd) AS perc_10,
        percentile_cont(0.34) WITHIN GROUP (ORDER BY hotel_per_room_usd) AS perc_34,
        percentile_cont(0.66) WITHIN GROUP (ORDER BY hotel_per_room_usd) AS perc_66,
        percentile_cont(0.90) WITHIN GROUP (ORDER BY hotel_per_room_usd) AS perc_90
    FROM hotel_costs
),
room_categories AS (
    SELECT
        hc.*,
        CASE
            WHEN hc.hotel_per_room_usd < p.perc_10 THEN 'Budget'
            WHEN hc.hotel_per_room_usd < p.perc_34 THEN 'Economy'
            WHEN hc.hotel_per_room_usd < p.perc_66 THEN 'Mid-range'
            WHEN hc.hotel_per_room_usd < p.perc_90 THEN 'Premium'
            ELSE 'Luxury'
        END AS hotel_price_category
    FROM hotel_costs hc
    CROSS JOIN percentiles p
)
SELECT *
FROM room_categories;



-- FLIGHTS TABLE
WITH flights_base AS (
    SELECT
        trip_id,
        origin_airport,
        destination_airport,
        seats,
        return_flight_booked,
        departure_time,
        return_time,
        trip_airline,
        checked_bags,
        base_fare_usd,
        return_time::DATE - departure_time::DATE AS travel_duration
    FROM flights
),
fare_percentiles AS (
    SELECT
        percentile_cont(0.10) WITHIN GROUP (ORDER BY base_fare_usd) AS perc_10,
        percentile_cont(0.34) WITHIN GROUP (ORDER BY base_fare_usd) AS perc_34,
        percentile_cont(0.66) WITHIN GROUP (ORDER BY base_fare_usd) AS perc_66,
        percentile_cont(0.90) WITHIN GROUP (ORDER BY base_fare_usd) AS perc_90
    FROM flights
),
flights_with_categories AS (
    SELECT
        f.*,
        CASE
            WHEN f.travel_duration < 2 THEN 'Day trip'
            WHEN f.travel_duration BETWEEN 2 AND 3 THEN 'Weekend getaway'
            WHEN f.travel_duration BETWEEN 4 AND 6 THEN 'Short vacation (4–6 days)'
            WHEN f.travel_duration BETWEEN 7 AND 10 THEN 'Standard vacation (7–10 days)'
            WHEN f.travel_duration BETWEEN 11 AND 14 THEN 'Extended vacation (11–14 days)'
            ELSE 'Long-term travel (15+ days)'
        END AS stay_category
    FROM flights_base f
),
flights_final AS (
    SELECT
        f.*,
        CASE
            WHEN f.base_fare_usd < p.perc_10 THEN 'Budget'
            WHEN f.base_fare_usd < p.perc_34 THEN 'Economy'
            WHEN f.base_fare_usd < p.perc_66 THEN 'Mid-range'
            WHEN f.base_fare_usd < p.perc_90 THEN 'Premium'
            ELSE 'Luxury'
        END AS flight_fare_category
    FROM flights_with_categories f
    CROSS JOIN fare_percentiles p
)
SELECT *
FROM flights_final;



-- SESSIONS TABLE
WITH user_session_counts AS (
    SELECT
        user_id,
        COUNT(session_id) AS total_sessions
    FROM sessions
    WHERE session_start > '2023-01-04'
    GROUP BY user_id
    HAVING COUNT(session_id) > 7
),
cte_sessions AS (
    SELECT
        s.session_id,
        s.user_id,
        s.trip_id,
        s.session_start,
        s.session_end,
        TRIM(TO_CHAR(s.session_start, 'Day')) AS session_start_weekday,
        EXTRACT(HOUR FROM s.session_start) AS session_start_hour,
        EXTRACT(EPOCH FROM (s.session_end - s.session_start)) / 60 AS session_duration_minutes,
        CASE
            WHEN s.flight_discount = FALSE AND s.hotel_discount = FALSE THEN 'no'
            WHEN s.flight_discount = TRUE AND s.hotel_discount = TRUE THEN 'full'
            ELSE 'partial'
        END AS discount,
        s.flight_discount,
        s.hotel_discount,
        s.flight_discount_amount,
        s.hotel_discount_amount,
        s.flight_booked,
        s.hotel_booked,
        s.page_clicks,
        s.cancellation
    FROM sessions s
    JOIN user_session_counts usc ON s.user_id = usc.user_id
    WHERE s.session_start > '2023-01-04'
),
percentile AS (
    SELECT
        percentile_cont(0.10) WITHIN GROUP (ORDER BY page_clicks) AS perc_10,
        percentile_cont(0.34) WITHIN GROUP (ORDER BY page_clicks) AS perc_34,
        percentile_cont(0.66) WITHIN GROUP (ORDER BY page_clicks) AS perc_66,
        percentile_cont(0.90) WITHIN GROUP (ORDER BY page_clicks) AS perc_90
    FROM sessions
),
page_clicks_category AS (
    SELECT
        session_id,
        user_id,
        page_clicks,
        CASE
            WHEN page_clicks <= p.perc_10 THEN 'Low activity'
            WHEN page_clicks <= p.perc_34 THEN 'Medium activity'
            WHEN page_clicks <= p.perc_66 THEN 'High activity'
            WHEN page_clicks > p.perc_90 THEN 'Very high activity'
            ELSE 'Unknown'
        END AS page_clicks_category
    FROM cte_sessions cte
    CROSS JOIN percentile p
)
SELECT
    cte.session_id,
    cte.user_id,
    cte.trip_id,
    cte.session_start,
    cte.session_end,
    cte.session_start_weekday,
    cte.session_start_hour,
    cte.session_duration_minutes,
    cte.discount,
    cte.flight_discount,
    cte.hotel_discount,
    cte.flight_discount_amount,
    cte.hotel_discount_amount,
    cte.flight_booked,
    cte.hotel_booked,
    cte.page_clicks,
    cte.cancellation,
    pcc.page_clicks_category
FROM cte_sessions cte
LEFT JOIN page_clicks_category pcc ON cte.session_id = pcc.session_id
ORDER BY cte.session_id;
