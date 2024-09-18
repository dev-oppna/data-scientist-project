import json
import requests
import hashlib

base_url = "https://api-stg.oppna.dev/v1/cardinals/opa/"

def get_enrichment(opa_id: str, name: str) -> dict:
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "date": "latest",
        "name": name
    })
    ses = requests.session()
    try:
        resp = ses.get(f"{base_url}{opa_id}", headers=headers, data=data)
        response = resp.json()
        return response
    except Exception as e:
        return {}
    
def generate_opa_id(phone:str) -> str:
    hash_phone = hashlib.sha1(str.encode(phone)).hexdigest()
    opa_id = "67762" + hash_phone + phone[-4:]
    return opa_id

MAPPING = {
    'Demographic': [
        'opa_id',
        'persona',
        'predicted_bank',
        'predicted_digital',
        'predicted_ecommerce_seller',
        'gender',
        'age',
        'gen_type',
        'district_registered',
        'city_registered',
        'province_registered',
        'district_domicile',
        'city_domicile',
        'province_domicile',
        'occupation',
        'marital_sts',
        'is_verified',
        'parents',
        'predicted_smartphone_owner',
        'predicted_luxury_phone_owner',
        'predicted_car_owner',
        'predicted_motorcycle_owner',
        'predicted_luxury_motorcycle_owner',
        'predicted_low_class_car_owner',
        'predicted_mid_class_car_owner',
        'predicted_high_class_car_owner',
        'predicted_smarttv_owner',
        'cctv_owner',
        'predicted_expensive_self_care_owner',
        'predicted_medium_self_care_owner',
        'predicted_high_class_computer_owner',
        'predicted_high_class_health_product',
        'predicted_high_class_household_owner',
        'social_commerce_user'],
    'Affinities': [
        'auto_geek',
        'digital_shopper',
        'sporty',
        'religious',
        'gadget_freak',
        'traveler',
        'fashionista',
        'beauty_enthusiast',
        'gamer',
        'visual_geek',
        'ever_insurance_user',
        'ever_loan_vehicle',
        'ever_take_cash_loan',
        'ever_take_loan',
        'health_conscious',
        'far_sighted_shopper',
        'gardener',
        'musician',
        'stationery_sorcerer',
        'pet_owner',
        'vaper',
        'book_lover',
        'home_cooking',
        'alcohol_drinker',
        'snack_sweets_lover',
        'soccer_fans',
        'badminton_lover',
        'basketball_lover',
        'tennis_lover',
        'workout_warrior',
        'home_organizer',
        'lock_safe_devotee',
        'midas_investor',
        'cinephile'],
    'Behavior': [
        'last_digital_presence',
        'engaged_shopper',
        'recently_traveled',
        'fintech_familiarity_normal_users',
        'fintech_familiarity_hidden_gems',
        'fintech_familiarity_fintech_literate',
        'fintech_familiarity_fintech_savvy',
        'digital_presence_chronic',
        'digital_presence_sick',
        'digital_presence_need_attention',
        'digital_presence_healthy',
        'cltv_monthly_12_months',
        'recently_has_child',
        'recently_decorate_renovate',
        'sexually_active',
        'morning_shopper',
        'noon_shopper',
        'night_shopper',
        'midnight_shopper',
        'weekday_shopper',
        'weekend_shopper',
        'payday_shopper',
        'early_month_shopper',
        'mid_month_shopper',
        'prefer_high_value_goods',
        'prefer_mid_to_high_value_goods',
        'prefer_mid_to_low_value_goods',
        'prefer_low_value_goods',
        'intensity',
        'digital_footprint',
        'momentum',
        'clumpiness']
    }