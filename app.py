from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import random
import math
import re
import json
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import sys
import os
import threading
from datetime import datetime
import uuid

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret")  # Change this in production

# Global variables to store datasets
food_df = None
restaurant_groups = None
hotel_df = None
poi_df = None
transport_df = None

# Global dictionary to store user sessions
user_sessions = {}

def load_datasets():
    """Load datasets once when the app starts"""
    global food_df, restaurant_groups, hotel_df, poi_df, transport_df
    
    print("📊 Loading and preprocessing datasets...")
    
    try:
        # 1. Food Reviews Dataset
        food_df = pd.read_csv('food_reviews_with_sentiment.csv')
        # Clean restaurant names and group by restaurant
        restaurant_groups = food_df.groupby('name of restaurant').agg({
            'positive_prob': 'mean',
            'priceINR': ['mean', 'min', 'max', 'count'],
            'location': 'first',
            'amenities_cuisine': 'first',
            'triptype': lambda x: x.value_counts().index[0],
            'reviewrating': 'mean'
        }).reset_index()

        # Flatten column names
        restaurant_groups.columns = ['restaurant_name', 'avg_sentiment', 'avg_price', 
                                    'min_price', 'max_price', 'review_count', 'location', 
                                    'cuisine', 'triptype', 'avg_rating']

        # Calculate price categories
        def categorize_price(price):
            if price < 300:
                return 'Budget'
            elif price < 700:
                return 'Moderate'
            elif price < 1500:
                return 'Expensive'
            else:
                return 'Premium'

        restaurant_groups['price_category'] = restaurant_groups['avg_price'].apply(categorize_price)
        restaurant_groups['restaurant_id'] = range(len(restaurant_groups))

        # 2. Hotel Rankings Dataset
        hotel_df = pd.read_csv('hotel_ranking_output.csv')
        # Clean hotel data
        hotel_df = hotel_df[['Hotel Name', 'StarRating', 'PriceINR', 'review_count', 'xgb_score', 'relevance_score']].copy()
        hotel_df.rename(columns={'Hotel Name': 'hotel_name'}, inplace=True)
        hotel_df['hotel_id'] = range(len(hotel_df))
        hotel_df['normalized_score'] = (hotel_df['xgb_score'] - hotel_df['xgb_score'].min()) / (hotel_df['xgb_score'].max() - hotel_df['xgb_score'].min())

        # 3. POI Rankings Dataset
        poi_df = pd.read_csv('poi_ranking_output.csv')
        poi_df.rename(columns={'Place Name': 'poi_name', 'Category': 'poi_category'}, inplace=True)
        poi_df['poi_id'] = range(len(poi_df))
        poi_df['normalized_xgb'] = (poi_df['xgb_score'] - poi_df['xgb_score'].min()) / (poi_df['xgb_score'].max() - poi_df['xgb_score'].min())

        # 4. Transportation Safety Dataset
        transport_df = pd.read_csv('safety_transport_iforest_results.csv')
        transport_df.rename(columns={'transport_name': 'transport_name'}, inplace=True)
        transport_df['transport_id'] = range(len(transport_df))
        transport_df['safety_score'] = transport_df['adjusted_reliability_score'] / 100  # Normalize to 0-1
        
        print("✅ Datasets loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return False

# Load datasets when app starts
load_datasets()

@dataclass
class UserPreferences:
    """User preferences and constraints"""
    budget: float
    days: int
    trip_type: str  # 'Family', 'Couple', 'Solo', 'Business', 'Group', 'Leisure'
    interests: List[str]
    cuisine_preferences: List[str]
    hotel_star_pref: float = 3.0
    meal_budget_per_day: float = 1000
    transport_safety_weight: float = 0.7  # 0-1, weight for safety vs time
    adventure_level: str = 'moderate'  # 'relaxed', 'moderate', 'intense'

@dataclass
class Activity:
    """Represents an activity in the itinerary"""
    type: str  # 'hotel', 'poi', 'restaurant', 'transport', 'leisure'
    name: str
    cost: float
    duration_hours: float
    time_slot: int  # 0=morning, 1=afternoon, 2=evening
    day: int
    metadata: Dict = field(default_factory=dict)

class TripChatbot:
    """NLP-powered chatbot for trip Q&A with basic intent classification"""
    
    def __init__(self, itinerary_state: Dict, user_prefs: UserPreferences):
        # Ensure itinerary_state has all required keys with safe defaults
        self.itinerary_state = {
            'selected_hotel': itinerary_state.get('selected_hotel', {}),
            'selected_pois': itinerary_state.get('selected_pois', []),
            'selected_restaurants': itinerary_state.get('selected_restaurants', []),
            'itinerary': itinerary_state.get('itinerary', defaultdict(list)),
            'total_cost': itinerary_state.get('total_cost', 0),
            'remaining_budget': itinerary_state.get('remaining_budget', 0)
        }
        self.user_prefs = user_prefs
        
        # Pre-calculate values for response formatting
        self._precalculate_values()
        
        # Define intents and their patterns
        self.intents = {
            'greeting': {
                'patterns': ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon'],
                'responses': [
                    "Hello! I'm your Trip Assistant. How can I help you with your itinerary?",
                    "Hi there! Ready to answer your questions about the trip!",
                    "Hey! Ask me anything about your itinerary!"
                ]
            },
            'budget': {
                'patterns': ['budget', 'cost', 'price', 'expensive', 'cheap', 'how much', 'total cost'],
                'responses': [
                    f"Your total budget is ₹{user_prefs.budget:,.0f}. Estimated cost is ₹{self.itinerary_state.get('total_cost', 0):,.0f}.",
                    f"Budget overview: Hotel: ~₹{self.hotel_total:,.0f}, Food: ~₹{self.food_total:,.0f}, Activities: ~₹{self.activities_total:,.0f}."
                ]
            },
            'hotel': {
                'patterns': ['hotel', 'accommodation', 'stay', 'where to stay', 'lodging'],
                'responses': [
                    f"Your hotel: {self.hotel_name}. {self.hotel_stars} Total cost: ₹{self.hotel_total_cost:,.0f} for {user_prefs.days} nights."
                ]
            },
            'restaurant': {
                'patterns': ['restaurant', 'food', 'eat', 'dining', 'meal', 'cuisine', 'lunch', 'dinner'],
                'responses': [
                    f"You have {self.restaurant_count} restaurants selected. Average meal cost: ₹{self.restaurant_mean:,.0f}."
                ]
            },
            'activities': {
                'patterns': ['activities', 'things to do', 'attractions', 'places to visit', 'poi', 'sightseeing'],
                'responses': [
                    f"You have {self.poi_count} attractions planned. Categories: {self.poi_categories}."
                ]
            },
            'timing': {
                'patterns': ['time', 'schedule', 'when', 'duration', 'how long', 'day', 'schedule'],
                'responses': [
                    f"Trip duration: {user_prefs.days} days. Each day has morning, afternoon, and evening slots.",
                    f"Daily schedule includes 2-3 main activities plus meals and leisure time."
                ]
            },
            'transport': {
                'patterns': ['transport', 'travel', 'commute', 'how to get', 'transportation', 'safety'],
                'responses': [
                    f"Transport safety weight: {user_prefs.transport_safety_weight:.1f} "
                    f"{'(Safety prioritized)' if user_prefs.transport_safety_weight > 0.7 else '(Balanced approach)'}.",
                    "Recommended transport modes: App-based taxis, pre-booked cars for safety."
                ]
            },
            'recommendation': {
                'patterns': ['recommend', 'suggest', 'best', 'top', 'popular', 'must see', 'must visit'],
                'responses': [
                    f"Based on your interests ({', '.join(user_prefs.interests)}), "
                    f"recommended categories: {self._get_recommendations()}."
                ]
            },
            'help': {
                'patterns': ['help', 'what can you do', 'assist', 'support', 'guide'],
                'responses': [
                    "I can help you with: budget details, hotel info, restaurant suggestions, "
                    "activity schedules, transport options, and general recommendations.",
                    "Ask me about: costs, timing, food, places to visit, or anything about your trip!"
                ]
            },
            'thank_you': {
                'patterns': ['thank', 'thanks', 'appreciate', 'grateful'],
                'responses': [
                    "You're welcome! Happy to help with your trip planning!",
                    "Glad I could assist! Enjoy your trip to Visakhapatnam!",
                    "My pleasure! Let me know if you have more questions."
                ]
            },
            'goodbye': {
                'patterns': ['bye', 'goodbye', 'exit', 'quit', 'see you', 'farewell'],
                'responses': [
                    "Goodbye! Have a wonderful trip to Visakhapatnam!",
                    "Safe travels! Enjoy your journey!",
                    "Bye! Don't hesitate to ask if you need more help."
                ]
            }
        }
        
        # FAQ database
        # FAQ database
        self.faqs = {
    'What is the best time to visit Visakhapatnam?': 
        "October to March is ideal with pleasant weather (20-30°C). Avoid monsoon (June-September).",
    
    'What are the must-visit attractions?': 
        "RK Beach, Kailasagiri, Borra Caves, Araku Valley, Submarine Museum, and Simhachalam Temple.",
    
    'Is Visakhapatnam safe for tourists?': 
        "Yes, Visakhapatnam is generally safe. Stick to well-lit areas at night and use app-based taxis.",
    
    'What is the local cuisine like?': 
        "Famous for seafood, Andhra cuisine (spicy), and local snacks like Punugulu and Dosa.",
    
    'How is the public transportation?': 
        "City buses, autos, and taxis are available. App-based cabs are recommended for convenience.",
    
    'What should I pack for the trip?': 
        "Light cotton clothes, sunscreen, hat, comfortable shoes, and a light jacket for evenings.",
    
    'Are there any local customs I should know?': 
        "Remove shoes at temples, dress modestly at religious sites, and greet locals with 'Namaste'.",
    
    'What is the average cost per day?': 
        "Budget: ₹2,000-3,000, Moderate: ₹3,000-6,000, Luxury: ₹6,000+ per person per day.",
    
    'Can I use credit cards everywhere?': 
        "Major hotels and restaurants accept cards, but carry cash for local markets and transport.",
    
    'What languages are spoken?': 
        "Telugu is the main language, but Hindi and English are widely understood in tourist areas.",

    # ➕ NEW FAQs ADDED BELOW

    'How many days are enough for Visakhapatnam?':
        "2–3 days are enough to cover major city attractions. 4–5 days are ideal if including Araku Valley.",

    'Are food and dining prices expensive?':
        "Food is affordable. Local meals cost ₹150–300, mid-range restaurants ₹400–800 per meal.",

    'Do restaurants offer vegetarian options?':
        "Yes, most restaurants offer vegetarian and vegan-friendly Andhra and North Indian dishes.",

    'Is Visakhapatnam good for family trips?':
        "Yes, it is family-friendly with beaches, parks, museums, and safe sightseeing options.",

    'Is Visakhapatnam suitable for solo travelers?':
        "Yes, it’s safe and affordable with good connectivity and plenty of hostels and cafes.",

    'What are the best beaches to visit?':
        "RK Beach, Rushikonda Beach, Yarada Beach, and Lawson’s Bay are popular choices.",

    'How are restaurant recommendations generated?':
        "Restaurants are recommended using ratings, average price, cuisine type, and sentiment analysis.",

    'Are food prices estimated or exact?':
        "Food prices are estimated averages based on restaurant datasets and may vary slightly.",

    'Can I customize my itinerary later?':
        "Yes, you can restart the planner and customize budget, days, and trip type anytime.",

    'Does the itinerary include transport cost?':
        "Transport cost is estimated based on trip duration and local travel averages."
}

        
    
    def _precalculate_values(self):
        """Pre-calculate values to avoid formatting issues in f-strings"""
        # Hotel values
        self.hotel_name = self.itinerary_state.get('selected_hotel', {}).get('name', 'Not selected yet')
        self.hotel_rating = self.itinerary_state.get('selected_hotel', {}).get('rating', 0)
        self.hotel_stars = '⭐' * int(self.hotel_rating) if self.hotel_rating else ''
        self.hotel_total_cost = self.itinerary_state.get('selected_hotel', {}).get('total_cost', 0)
        
        # Calculate hotel total for budget overview
        self.hotel_total = self.itinerary_state.get('selected_hotel', {}).get('total_cost', 0) if self.itinerary_state.get('selected_hotel') else 0
        
        # Restaurant values
        self.restaurant_count = len(self.itinerary_state.get('selected_restaurants', []))
        if self.restaurant_count > 0:
            prices = [float(r.get('avg_price', 0)) for r in self.itinerary_state.get('selected_restaurants', [])]
            self.restaurant_mean = np.mean(prices) if prices else 0
        else:
            self.restaurant_mean = 0
            
        # Calculate food total for budget overview
        self.food_total = sum(float(r.get('avg_price', 0)) for r in self.itinerary_state.get('selected_restaurants', []))
        
        # POI values
        self.poi_count = len(self.itinerary_state.get('selected_pois', []))
        poi_categories_set = set()
        for poi in self.itinerary_state.get('selected_pois', []):
            category = poi.get('poi_category', '')
            if category:
                poi_categories_set.add(category)
        self.poi_categories = ', '.join(list(poi_categories_set)) if poi_categories_set else 'Various'
        
        # Calculate activities total for budget overview
        self.activities_total = sum(float(p.get('Entry_fee_numeric', 0)) for p in self.itinerary_state.get('selected_pois', []))
    
    def _get_recommendations(self) -> str:
        """Get personalized recommendations based on user interests"""
        recommendations = []
        
        if 'Beach' in self.user_prefs.interests:
            recommendations.append("RK Beach, Rushikonda Beach, Bheemili Beach")
        if 'Historical' in self.user_prefs.interests:
            recommendations.append("Submarine Museum, Victory at Sea Memorial")
        if 'Religious' in self.user_prefs.interests:
            recommendations.append("Simhachalam Temple, Sri Venkateswara Temple")
        if 'Nature' in self.user_prefs.interests:
            recommendations.append("Kailasagiri, Indira Gandhi Zoological Park")
        
        return ', '.join(recommendations) if recommendations else "Check the popular attractions list"
    
    def classify_intent(self, user_input: str) -> str:
        """Basic intent classification using keyword matching"""
        user_input = user_input.lower().strip()
        
        # Check for FAQ questions
        for question in self.faqs.keys():
            if any(word in user_input for word in question.lower().split()[:3]):
                return 'faq'
        
        # Check intents
        best_intent = 'unknown'
        max_matches = 0
        
        for intent, data in self.intents.items():
            matches = sum(1 for pattern in data['patterns'] if pattern in user_input)
            if matches > max_matches:
                max_matches = matches
                best_intent = intent
        
        return best_intent if max_matches > 0 else 'unknown'
    
    def extract_entities(self, user_input: str) -> Dict:
        """Extract basic entities from user input"""
        entities = {
            'day': None,
            'time': None,
            'activity': None,
            'cost': None
        }
        
        # Extract day number
        day_match = re.search(r'day\s*(\d+)', user_input.lower())
        if day_match:
            try:
                entities['day'] = int(day_match.group(1))
            except:
                entities['day'] = None
        
        # Extract time of day
        if 'morning' in user_input.lower():
            entities['time'] = 'morning'
        elif 'afternoon' in user_input.lower():
            entities['time'] = 'afternoon'
        elif 'evening' in user_input.lower():
            entities['time'] = 'evening'
        
        # Extract cost-related words
        cost_words = ['cost', 'price', 'budget', 'expensive', 'cheap']
        for word in cost_words:
            if word in user_input.lower():
                entities['cost'] = True
                break
        
        return entities
    
    def get_faq_answer(self, user_input: str) -> Optional[str]:
        """Get answer from FAQ database"""
        user_input = user_input.lower()
        
        # Find the most relevant FAQ question
        best_match = None
        best_score = 0
        
        for question in self.faqs.keys():
            question_words = set(question.lower().split())
            input_words = set(user_input.split())
            
            # Simple word overlap scoring
            overlap = len(question_words.intersection(input_words))
            score = overlap / max(len(question_words), 1)
            
            if score > best_score and score > 0.3:
                best_score = score
                best_match = question
        
        if best_match:
            return f"Q: {best_match}\nA: {self.faqs[best_match]}"
        
        return None
    
    def get_day_schedule(self, day: int) -> str:
        """Get detailed schedule for a specific day"""
        if day < 1 or day > self.user_prefs.days:
            return f"Invalid day. Your trip is {self.user_prefs.days} days long."
        
        activities = self.itinerary_state['itinerary'].get(day-1, [])
        if not activities:
            return f"No activities scheduled for Day {day} yet."
        
        schedule = f"📅 Schedule for Day {day}:\n"
        time_slots = {0: "🌅 Morning", 1: "☀️ Afternoon", 2: "🌇 Evening"}
        
        for activity in activities:
            if hasattr(activity, 'time_slot'):
                time_slot = time_slots.get(activity.time_slot, "Flexible Time")
            else:
                time_slot = "Scheduled"
                
            if hasattr(activity, 'type'):
                if activity.type == 'poi':
                    schedule += f"{time_slot}: Visit {activity.name}\n"
                elif activity.type == 'restaurant':
                    meal_time = activity.metadata.get('meal_time', 'Meal') if hasattr(activity, 'metadata') else 'Meal'
                    schedule += f"{time_slot}: {meal_time} at {activity.name}\n"
                elif activity.type == 'leisure':
                    schedule += f"{time_slot}: {activity.name}\n"
                elif activity.type == 'hotel':
                    schedule += f"{time_slot}: Check-in at {activity.name}\n"
            else:
                schedule += f"{time_slot}: {activity}\n"
        
        return schedule
    
    def generate_response(self, user_input: str) -> str:
        """Generate response based on intent and entities"""
        intent = self.classify_intent(user_input)
        entities = self.extract_entities(user_input)
        
        # Check FAQ first
        faq_answer = self.get_faq_answer(user_input)
        if faq_answer:
            return faq_answer
        
        # Check for specific day query
        if entities['day']:
            return self.get_day_schedule(entities['day'])
        
        # Generate response based on intent
        if intent in self.intents:
            try:
                response = random.choice(self.intents[intent]['responses'])
                
                # Add personalized details based on intent
                if intent == 'budget':
                    remaining = self.itinerary_state.get('remaining_budget', 0)
                    response += f" Remaining budget: ₹{remaining:,.0f}."
                
                elif intent == 'restaurant':
                    if self.restaurant_count > 0:
                        cuisines = set(r.get('cuisine', '') for r in self.itinerary_state.get('selected_restaurants', []))
                        cuisines = [c for c in cuisines if c]  # Filter out empty strings
                        if cuisines:
                            response += f" Cuisines: {', '.join(list(cuisines)[:3])}."
                
                elif intent == 'activities':
                    if self.poi_count > 0:
                        response += " Check the detailed itinerary for timings."
                
                return response
            except Exception as e:
                return f"I'm having trouble accessing that information. Error: {str(e)[:50]}..."
        
        # Default response for unknown intent
        return "I'm not sure I understand. You can ask me about:\n" \
               "- Budget and costs\n- Hotel information\n- Restaurant suggestions\n" \
               "- Daily schedules (e.g., 'What's on day 2?')\n" \
               "- Activity recommendations\n- Transport options\n" \
               "Or ask 'help' for more options."

class RLItineraryEnvironment:
    """RL Environment for itinerary planning"""
    
    def __init__(self, user_prefs: UserPreferences):
        self.user_prefs = user_prefs
        self.state = None
        self.available_hotels = []
        self.available_pois = []
        self.available_restaurants = []
        self.available_transport = []
        
        # Initialize data
        self._initialize_data()
        self.reset()
        
    def _initialize_data(self):
        """Initialize and filter data based on user preferences"""
        
        # Filter hotels by budget and star rating
        max_hotel_per_night = self.user_prefs.budget * 0.4 / self.user_prefs.days
        self.available_hotels = hotel_df[
            (hotel_df['PriceINR'] <= max_hotel_per_night) &
            (hotel_df['StarRating'] >= self.user_prefs.hotel_star_pref)
        ].copy()
        
        if len(self.available_hotels) == 0:
            # Relax constraints if no hotels found
            self.available_hotels = hotel_df[hotel_df['StarRating'] >= 3.0].copy()
        
        # Sort hotels by ranking score
        self.available_hotels = self.available_hotels.sort_values('xgb_score', ascending=False)
        
        # Filter POIs by interests
        interest_categories = {
            'beach': ['Beach'],
            'historical': ['War Memorial', 'Historical Museum', 'Maritime Museum'],
            'religious': ['Hindu Temple'],
            'adventure': ['Wildlife Sanctuary', 'Hill Station', 'Scenic Viewpoint'],
            'nature': ['Wildlife Sanctuary', 'Scenic Park'],
            'family': ['Aquarium', 'Zoo', 'Park'],
            'culture': ['Historical Museum', 'Maritime Museum'],
            'scenic': ['Scenic Park', 'Viewpoint', 'Lighthouse', 'Hilltop Park']
        }
        
        poi_categories = []
        for interest in self.user_prefs.interests:
            if interest.lower() in interest_categories:
                poi_categories.extend(interest_categories[interest.lower()])
        
        if poi_categories:
            self.available_pois = poi_df[poi_df['poi_category'].isin(poi_categories)].copy()
        else:
            self.available_pois = poi_df.copy()
        
        # Sort POIs by ranking
        self.available_pois = self.available_pois.sort_values('final_rank')
        
        # Enhanced restaurant filtering with price constraints
        daily_meal_budget = self.user_prefs.meal_budget_per_day
        max_per_meal = daily_meal_budget / 2  # Assuming 2 main meals per day
        
        # Filter by cuisine and trip type
        restaurant_mask = restaurant_groups['triptype'] == self.user_prefs.trip_type
        cuisine_mask = False
        for cuisine in self.user_prefs.cuisine_preferences:
            cuisine_mask = cuisine_mask | restaurant_groups['cuisine'].str.contains(cuisine, case=False, na=False)
        
        # Filter by price based on budget
        price_mask = restaurant_groups['avg_price'] <= max_per_meal
        
        # Combine all masks
        self.available_restaurants = restaurant_groups[
            restaurant_mask & cuisine_mask & price_mask
        ].copy()
        
        # If no restaurants found, relax price constraint
        if len(self.available_restaurants) == 0:
            self.available_restaurants = restaurant_groups[
                restaurant_mask & cuisine_mask &
                (restaurant_groups['avg_price'] <= max_per_meal * 1.5)
            ].copy()
        
        # If still none, fallback to all restaurants with good sentiment
        if len(self.available_restaurants) == 0:
            self.available_restaurants = restaurant_groups[
                (restaurant_groups['avg_sentiment'] > 0.7) &
                (restaurant_groups['avg_price'] <= daily_meal_budget)
            ].copy()
        
        # Calculate composite score for ranking
        self.available_restaurants['composite_score'] = (
            self.available_restaurants['avg_sentiment'] * 0.4 +
            (5 - self.available_restaurants['avg_price'] / 200) * 0.3 +  # Lower price = higher score
            self.available_restaurants['avg_rating'] / 5 * 0.3
        )
        
        self.available_restaurants = self.available_restaurants.sort_values('composite_score', ascending=False)
        
        # Filter transport by safety
        if self.user_prefs.transport_safety_weight > 0.7:
            self.available_transport = transport_df[
                transport_df['risk_level'].isin(['Low', 'Medium'])
            ].copy()
        else:
            self.available_transport = transport_df.copy()
        
        # Sort transport by composite score
        self.available_transport['composite_score'] = (
            self.available_transport['safety_score'] * self.user_prefs.transport_safety_weight +
            (100 - self.available_transport['scheduled_travel_minutes'] / 10) * (1 - self.user_prefs.transport_safety_weight)
        )
        self.available_transport = self.available_transport.sort_values('composite_score', ascending=False)
    
    def reset(self):
        """Reset environment to initial state"""
        self.state = {
            'current_day': 0,
            'current_time_slot': 0,  # 0=morning, 1=afternoon, 2=evening
            'remaining_budget': self.user_prefs.budget,
            'remaining_time': self.user_prefs.days * 3,  # 3 time slots per day
            'selected_hotel': None,
            'selected_pois': [],
            'selected_restaurants': [],
            'selected_transport': [],
            'itinerary': defaultdict(list),
            'day_wise_food_spend': defaultdict(float),
            'total_cost': 0,
            'reward_accumulated': 0,
            'hotel_booked': False
        }
        return self._get_state_representation()
    
    def _get_state_representation(self):
        """Convert state to numerical representation for RL agent"""
        state_vec = [
            self.state['current_day'] / self.user_prefs.days,
            self.state['current_time_slot'] / 2,
            self.state['remaining_budget'] / self.user_prefs.budget,
            len(self.state['selected_pois']) / (self.user_prefs.days * 2),  # Max 2 POIs per day
            len(self.state['selected_restaurants']) / (self.user_prefs.days * 2),  # Max 2 meals per day
            1.0 if self.state['hotel_booked'] else 0.0,
            self.state['remaining_time'] / (self.user_prefs.days * 3)
        ]
        return np.array(state_vec, dtype=np.float32)
    
    def get_available_actions(self) -> List[Dict]:
        """Get list of available actions in current state"""
        actions = []
        day = self.state['current_day']
        time_slot = self.state['current_time_slot']
        
        # Check if we can still add activities for current day
        if len(self.state['itinerary'][day]) >= 4:  # Max 4 activities per day
            return []
        
        # Action: Select hotel (only once, in morning of first day)
        if not self.state['hotel_booked'] and day == 0 and time_slot == 0:
            for i, hotel in self.available_hotels.head(5).iterrows():
                actions.append({
                    'type': 'select_hotel',
                    'id': int(hotel['hotel_id']),
                    'name': hotel['hotel_name'],
                    'cost': float(hotel['PriceINR']) * self.user_prefs.days
                })
        
        # Action: Select POI (morning or afternoon)
        if time_slot in [0, 1] and len([a for a in self.state['itinerary'][day] if a.type == 'poi']) < 2:
            for i, poi in self.available_pois.head(8).iterrows():
                actions.append({
                    'type': 'select_poi',
                    'id': int(poi['poi_id']),
                    'name': poi['poi_name'],
                    'cost': float(poi['Entry_fee_numeric']),
                    'category': poi['poi_category']
                })
        
        # Action: Select restaurant (afternoon or evening)
        if time_slot in [1, 2] and len([a for a in self.state['itinerary'][day] if a.type == 'restaurant']) < 2:
            for i, restaurant in self.available_restaurants.head(6).iterrows():
                actions.append({
                    'type': 'select_restaurant',
                    'id': int(restaurant['restaurant_id']),
                    'name': restaurant['restaurant_name'],
                    'cost': float(restaurant['avg_price']),
                    'cuisine': restaurant['cuisine'],
                    'min_price': float(restaurant['min_price']),
                    'max_price': float(restaurant['max_price'])
                })
        
        # Action: Add leisure time
        if len(self.state['itinerary'][day]) < 4:
            actions.append({
                'type': 'add_leisure',
                'name': 'Free Time / Rest',
                'cost': 0.0
            })
        
        # Action: Move to next time slot
        if len(self.state['itinerary'][day]) > 0:
            actions.append({
                'type': 'next_slot'
            })
        
        # Action: End day (if we have at least 2 activities)
        if len(self.state['itinerary'][day]) >= 2:
            actions.append({
                'type': 'end_day'
            })
        
        return actions
    
    def step(self, action: Dict):
        """Execute action and return new state, reward, done"""
        reward = 0
        done = False
        
        # Execute action based on type
        if action['type'] == 'select_hotel':
            hotel = self.available_hotels[self.available_hotels['hotel_id'] == action['id']].iloc[0]
            cost = action['cost']
            
            if cost <= self.state['remaining_budget']:
                self.state['selected_hotel'] = {
                    'name': hotel['hotel_name'],
                    'price_per_night': float(hotel['PriceINR']),
                    'total_cost': cost,
                    'rating': float(hotel['StarRating']),
                    'score': float(hotel['xgb_score'])
                }
                self.state['remaining_budget'] -= cost
                self.state['total_cost'] += cost
                self.state['hotel_booked'] = True
                
                # Add to itinerary
                activity = Activity(
                    type='hotel',
                    name=hotel['hotel_name'],
                    cost=cost,
                    duration_hours=0,  # Continuous stay
                    time_slot=self.state['current_time_slot'],
                    day=self.state['current_day'],
                    metadata={
                        'rating': float(hotel['StarRating']),
                        'price_per_night': float(hotel['PriceINR'])
                    }
                )
                self.state['itinerary'][self.state['current_day']].append(activity)
                
                # Calculate reward
                reward = self._calculate_hotel_reward(hotel, cost)
        
        elif action['type'] == 'select_poi':
            poi = self.available_pois[self.available_pois['poi_id'] == action['id']].iloc[0]
            self.state['selected_pois'].append(poi.to_dict())
            cost = action['cost']
            
            if cost <= self.state['remaining_budget']:
                self.state['selected_pois'].append(poi.to_dict())
                self.state['remaining_budget'] -= cost
                self.state['total_cost'] += cost
                
                # Add to itinerary
                activity = Activity(
                    type='poi',
                    name=poi['poi_name'],
                    cost=cost,
                    duration_hours=2.0,  # Average visit duration
                    time_slot=self.state['current_time_slot'],
                    day=self.state['current_day'],
                    metadata={
                        'category': poi['poi_category'],
                        'distance': float(poi['Distance_numeric']),
                        'xgb_score': float(poi['xgb_score'])
                    }
                )
                self.state['itinerary'][self.state['current_day']].append(activity)
                
                # Calculate reward
                reward = self._calculate_poi_reward(poi, cost)
        
        elif action['type'] == 'select_restaurant':
            restaurant = self.available_restaurants[
                self.available_restaurants['restaurant_id'] == action['id']
            ].iloc[0]

            cost = action['cost']
            day = self.state['current_day']

            if cost <= self.state['remaining_budget']:
        # Keep ONE append (remove duplicate)
                self.state['selected_restaurants'].append(restaurant.to_dict())

        # ✅ ADD THIS LINE (THIS FIXES ₹0 FOOD COST)
                self.state['day_wise_food_spend'][day] += cost

                self.state['remaining_budget'] -= cost
                self.state['total_cost'] += cost

                meal_time = 'Lunch' if self.state['current_time_slot'] == 1 else 'Dinner'

                activity = Activity(
                    type='restaurant',
                    name=restaurant['restaurant_name'],
                    cost=cost,
                    duration_hours=1.5,
                    time_slot=self.state['current_time_slot'],
                    day=day,
                metadata={
                    'cuisine': restaurant['cuisine'],
                    'avg_price': float(restaurant['avg_price']),
                    'sentiment': float(restaurant['avg_sentiment']),
                    'rating': float(restaurant['avg_rating']),
                    'meal_time': meal_time
                }
                )

                self.state['itinerary'][day].append(activity)

                reward = self._calculate_restaurant_reward(restaurant, cost)
        
        elif action['type'] == 'add_leisure':
            # Add leisure time
            activity = Activity(
                type='leisure',
                name='Free Time / Rest',
                cost=0.0,
                duration_hours=1.0,
                time_slot=self.state['current_time_slot'],
                day=self.state['current_day'],
                metadata={'description': 'Relaxation or exploration time'}
            )
            self.state['itinerary'][self.state['current_day']].append(activity)
            reward = 0.5  # Small reward for balanced schedule
        
        elif action['type'] == 'next_slot':
            # Move to next time slot
            self.state['current_time_slot'] += 1
            if self.state['current_time_slot'] > 2:  # End of day
                self.state['current_time_slot'] = 0
                self.state['current_day'] += 1
            reward = 0.1  # Small reward for progressing
        
        elif action['type'] == 'end_day':
            # End current day
            self.state['current_day'] += 1
            self.state['current_time_slot'] = 0
            reward = 1.0  # Reward for completing a day
            
            # Check if all days completed
            if self.state['current_day'] >= self.user_prefs.days:
                done = True
                # Add final completion reward
                reward += self._calculate_completion_reward()
        
        # Update remaining time
        self.state['remaining_time'] -= 1
        
        # Check termination conditions
        if self.state['remaining_budget'] <= 0 or self.state['remaining_time'] <= 0:
            done = True
            reward += self._calculate_completion_reward()
        
        self.state['reward_accumulated'] += reward
        
        return self._get_state_representation(), reward, done, self.state
    
    # ========== REWARD FUNCTIONS ==========
    
    def _calculate_hotel_reward(self, hotel, cost) -> float:
        """Reward for selecting a hotel"""
        reward = 0
        
        # Base reward from hotel ranking score
        reward += hotel['xgb_score'] * 2
        
        # Budget efficiency reward
        budget_ratio = self.user_prefs.budget / (cost + 1)
        reward += min(budget_ratio * 0.5, 3)
        
        # Star rating reward
        if hotel['StarRating'] >= 4.5:
            reward += 2
        elif hotel['StarRating'] >= 4.0:
            reward += 1
        
        # Review count reward
        if hotel['review_count'] > 50:
            reward += 1
        
        return reward
    
    def _calculate_poi_reward(self, poi, cost) -> float:
        """Reward for selecting a POI"""
        reward = 0
        
        # Base reward from POI ranking
        rank_reward = (21 - poi['final_rank']) * 0.5  # Higher rank = higher reward
        reward += rank_reward
        
        # Distance penalty (closer is better)
        distance_penalty = poi['Distance_numeric'] * 0.1
        reward -= distance_penalty
        
        # Entry fee consideration
        if poi['Entry_fee_numeric'] > 100:
            reward -= 0.5
        elif poi['Entry_fee_numeric'] == 0:
            reward += 1  # Bonus for free attractions
        
        # Diversity reward (avoid repeating same category)
        selected_categories = [p['poi_category'] for p in self.state['selected_pois']]
        if poi['poi_category'] not in selected_categories:
            reward += 1.5
        
        # Interest matching reward
        for interest in self.user_prefs.interests:
            if interest.lower() in poi['poi_category'].lower():
                reward += 1
        
        return reward
    
    def _calculate_restaurant_reward(self, restaurant, cost) -> float:
        """Enhanced reward for selecting a restaurant"""
        reward = 0
        
        # Sentiment reward (most important)
        reward += restaurant['avg_sentiment'] * 12
        
        # Price efficiency reward - more nuanced
        price_ratio = self.user_prefs.meal_budget_per_day / (cost + 1)
        if price_ratio > 2:
            reward += 3  # Very budget-friendly
        elif price_ratio > 1.5:
            reward += 2  # Moderately budget-friendly
        elif price_ratio > 1:
            reward += 1  # Within budget
        
        # Budget penalty if too expensive
        if cost > self.user_prefs.meal_budget_per_day * 1.5:
            reward -= 2
        
        # Cuisine preference match
        for cuisine in self.user_prefs.cuisine_preferences:
            if cuisine.lower() in str(restaurant['cuisine']).lower():
                reward += 3  # Increased reward for cuisine match
        
        # Trip type match
        if restaurant['triptype'] == self.user_prefs.trip_type:
            reward += 2
        
        # Rating reward with more granularity
        if restaurant['avg_rating'] >= 4.5:
            reward += 2.5
        elif restaurant['avg_rating'] >= 4.0:
            reward += 1.5
        elif restaurant['avg_rating'] >= 3.5:
            reward += 0.5
        
        # Review count reward
        if restaurant['review_count'] > 100:
            reward += 1.5
        elif restaurant['review_count'] > 50:
            reward += 1
        
        # Diversity reward (avoid same cuisine repeatedly)
        selected_cuisines = [r['cuisine'] for r in self.state['selected_restaurants']]
        if restaurant['cuisine'] not in selected_cuisines:
            reward += 2
        
        return reward
    
    def _calculate_completion_reward(self) -> float:
        """Final reward for completing itinerary"""
        reward = 0
        
        # Completeness bonus
        poi_count = len(self.state['selected_pois'])
        restaurant_count = len(self.state['selected_restaurants'])
        
        expected_pois = self.user_prefs.days * 1.5
        expected_restaurants = self.user_prefs.days * 2
        
        if poi_count >= expected_pois:
            reward += 5
        if restaurant_count >= expected_restaurants:
            reward += 5
        
        # Budget efficiency bonus
        budget_used = self.user_prefs.budget - self.state['remaining_budget']
        if budget_used > 0:
            efficiency = (self.user_prefs.budget / budget_used) * 3
            reward += min(efficiency, 10)
        
        # Diversity bonus
        if poi_count > 0:
            poi_categories = set([p['poi_category'] for p in self.state['selected_pois']])
            reward += len(poi_categories) * 1.5
        
        # Hotel selection bonus
        if self.state['hotel_booked']:
            reward += 3
        
        # Balanced schedule bonus
        for day in range(self.user_prefs.days):
            day_activities = len(self.state['itinerary'][day])
            if 2 <= day_activities <= 4:
                reward += 2
        
        return reward

class QLearningAgent:
    """Q-Learning agent for itinerary planning"""
    
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 0.1, 
                 discount_factor: float = 0.95,
                 exploration_rate: float = 0.3,
                 exploration_decay: float = 0.995):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        
        # Initialize Q-table
        self.q_table = np.zeros((state_size, action_size))
        
    def get_action(self, state: np.ndarray, available_actions: List[Dict]) -> int:
        """Choose action using epsilon-greedy policy"""
        if len(available_actions) == 0:
            return -1
        
        if random.random() < self.exploration_rate:
            # Explore: random action
            return random.randint(0, len(available_actions) - 1)
        else:
            # Exploit: best action based on Q-values
            # We need to map available actions to Q-table indices
            state_idx = self._state_to_index(state)
            q_values = self.q_table[state_idx, :len(available_actions)]
            return np.argmax(q_values)
    
    def _state_to_index(self, state: np.ndarray) -> int:
        """Convert continuous state to discrete index"""
        # Simple discretization for demo (in practice, use function approximation)
        discrete_state = []
        for i, val in enumerate(state):
            # Discretize each dimension into 3 bins
            if val < 0.33:
                discrete_state.append(0)
            elif val < 0.66:
                discrete_state.append(1)
            else:
                discrete_state.append(2)
        
        # Convert to single index (3^7 possible states = 2187)
        index = 0
        for i, val in enumerate(discrete_state):
            index += val * (3 ** i)
        
        return min(index, self.state_size - 1)
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, next_available_actions: List[Dict], done: bool):
        """Update Q-table using Q-learning"""
        state_idx = self._state_to_index(state)
        next_state_idx = self._state_to_index(next_state)
        
        # Current Q-value
        current_q = self.q_table[state_idx, action]
        
        if done:
            # Terminal state
            target_q = reward
        else:
            # Max Q-value for next state
            if next_available_actions:
                next_q_values = self.q_table[next_state_idx, :len(next_available_actions)]
                max_next_q = np.max(next_q_values)
            else:
                max_next_q = 0
            
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state_idx, action] = current_q + self.learning_rate * (target_q - current_q)
        
        # Decay exploration rate
        self.exploration_rate *= self.exploration_decay
    
    def train(self, env: RLItineraryEnvironment, episodes: int = 1000):
        """Train the agent"""
        print(f"🧠 Training RL agent for {episodes} episodes...")
        
        rewards_history = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                available_actions = env.get_available_actions()
                if not available_actions:
                    break
                
                action_idx = self.get_action(state, available_actions)
                if action_idx == -1:
                    break
                
                action = available_actions[action_idx]
                next_state, reward, done, _ = env.step(action)
                next_available_actions = env.get_available_actions()
                
                self.update(state, action_idx, reward, next_state, next_available_actions, done)
                
                state = next_state
                total_reward += reward
            
            rewards_history.append(total_reward)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"  Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Exploration: {self.exploration_rate:.3f}")
        
        return rewards_history
    
    def generate_itinerary(self, env: RLItineraryEnvironment) -> Dict:
        """Generate itinerary using trained policy"""
        # Disable exploration
        original_exploration = self.exploration_rate
        self.exploration_rate = 0.01
        
        state = env.reset()
        done = False
        
        while not done:
            available_actions = env.get_available_actions()
            if not available_actions:
                break
            
            action_idx = self.get_action(state, available_actions)
            if action_idx == -1:
                break
            
            action = available_actions[action_idx]
            state, reward, done, env_state = env.step(action)
        
        # Restore exploration rate
        self.exploration_rate = original_exploration
        
        return env_state
    
    def enhance_day_wise_itinerary(itinerary_state, user_prefs):
        """
        Human-centric post-processing layer to improve satisfaction
        """
        enhanced = defaultdict(list)

        for day in range(user_prefs.days):
            day_acts = itinerary_state['itinerary'].get(day, [])

            # Separate activities
            pois = [a for a in day_acts if a.type == 'poi']
            restaurants = [a for a in day_acts if a.type == 'restaurant']
            hotel = [a for a in day_acts if a.type == 'hotel']
            leisure = [a for a in day_acts if a.type == 'leisure']

            # ---- MORNING ----
            if hotel:
                enhanced[day].append(hotel[0])

            if pois:
                enhanced[day].append(pois[0])

            # ---- LUNCH ----
            lunch = next((r for r in restaurants if r.metadata.get('meal_time') == 'Lunch'), None)
            if lunch:
                enhanced[day].append(lunch)

            # ---- AFTERNOON ----
            if len(pois) > 1:
                enhanced[day].append(pois[1])
            else:
                enhanced[day].append(Activity(
                    type='leisure',
                    name='Relax / Local Exploration',
                    cost=0,
                    duration_hours=1.0,
                    time_slot=1,
                    day=day
                ))

            # ---- EVENING ----
            enhanced[day].append(Activity(
                type='leisure',
                name='Evening Walk / Shopping',
                cost=0,
                duration_hours=1.0,
                time_slot=2,
                day=day
            ))

            # ---- DINNER ----
            dinner = next((r for r in restaurants if r.metadata.get('meal_time') == 'Dinner'), None)
            if dinner:
                enhanced[day].append(dinner)

        itinerary_state['itinerary'] = enhanced
        return itinerary_state


def format_itinerary_for_html(itinerary_state: Dict, user_prefs: UserPreferences) -> Dict:
    """Format itinerary for HTML display with cost breakdown"""
    
    itinerary_data = {
        'trip_summary': {},
        'hotel': {},
        'days': [],
        'statistics': {},
        'cost_breakdown': {},
        'restaurants': [],
        'tips': []
    }
    
    # Trip Summary
    itinerary_data['trip_summary'] = {
        'trip_type': user_prefs.trip_type,
        'days': user_prefs.days,
        'budget': f"₹{user_prefs.budget:,.0f}",
        'estimated_cost': f"₹{itinerary_state.get('total_cost', 0):,.0f}",
        'remaining_budget': f"₹{itinerary_state.get('remaining_budget', 0):,.0f}",
        'utilization': f"{(itinerary_state.get('total_cost', 0) / user_prefs.budget * 100):.1f}%" if user_prefs.budget > 0 else "0%"
    }
    
    # Hotel Information
    if itinerary_state.get('selected_hotel'):
        hotel = itinerary_state['selected_hotel']
        itinerary_data['hotel'] = {
            'name': hotel.get('name', 'Unknown'),
            'rating': hotel.get('rating', 0),
            'stars': '⭐' * int(hotel.get('rating', 0)),
            'price_per_night': f"₹{hotel.get('price_per_night', 0):,.0f}",
            'total_cost': f"₹{hotel.get('total_cost', 0):,.0f}",
            'score': f"{hotel.get('score', 0):.2f}"
        }
    else:
        itinerary_data['hotel'] = {
            'name': 'Not selected (budget constraints)',
            'rating': 0,
            'stars': ''
        }
    
    # Day-wise Itinerary
    time_slots = {
        0: "🌅 Morning (9:00 AM - 12:00 PM)",
        1: "☀️ Afternoon (12:00 PM - 4:00 PM)",
        2: "🌇 Evening (4:00 PM - 8:00 PM)",
        3: "🌃 Night (8:00 PM onwards)"
    }
    
    for day in range(user_prefs.days):
        day_data = {
            'day_number': day + 1,
            'activities': []
        }
        
        activities = itinerary_state.get('itinerary', defaultdict(list))[day]
        if not activities:
            day_data['activities'].append({
                'time': 'No activities',
                'type': 'empty',
                'name': 'No activities scheduled'
            })
            itinerary_data['days'].append(day_data)
            continue
        
        # Sort activities by time slot
        activities_sorted = sorted(activities, key=lambda x: getattr(x, 'time_slot', 0))
        
        for activity in activities_sorted:
            time_slot = time_slots.get(getattr(activity, 'type', '') == 'hotel' and 0 or getattr(activity, 'time_slot', 0), "Flexible Time")
            
            activity_data = {
                'time': time_slot,
                'name': activity.name,
                'type': getattr(activity, 'type', '')
            }
            
            if activity.type == 'hotel':
                activity_data.update({
                    'icon': '🏨',
                    'details': [
                        f"Check-in at {activity.name}",
                        f"Rating: {activity.metadata.get('rating', 'N/A')} stars",
                        f"Total cost: ₹{activity.cost:,.0f}"
                    ]
                })
            
            elif activity.type == 'poi':
                activity_data.update({
                    'icon': '🏛️',
                    'details': [
                        f"Visit {activity.name}",
                        f"Category: {activity.metadata.get('category', 'Attraction')}",
                        f"Duration: {activity.duration_hours} hours",
                        f"Entry Fee: {'Free' if activity.cost == 0 else f'₹{activity.cost:,.0f}'}",
                        f"Score: {activity.metadata.get('xgb_score', 'N/A'):.2f}"
                    ]
                })
            
            elif activity.type == 'restaurant':
                meal_icon = '🍽️' if activity.metadata.get('meal_time') == 'Lunch' else '🍴'
                activity_data.update({
                    'icon': meal_icon,
                    'details': [
                        f"{activity.metadata.get('meal_time', 'Meal')} at {activity.name}",
                        f"Cuisine: {activity.metadata.get('cuisine', 'Multi-cuisine')}",
                        f"Price Range: ₹{activity.metadata.get('min_price', 0):,.0f} - ₹{activity.metadata.get('max_price', 0):,.0f}",
                        f"Average Cost: ₹{activity.metadata.get('avg_price', 0):,.0f}",
                        f"Duration: {activity.duration_hours} hours",
                        f"Rating: {activity.metadata.get('rating', 0):.1f}/5",
                        f"Sentiment Score: {activity.metadata.get('sentiment', 0):.2f}"
                    ]
                })
            
            elif activity.type == 'leisure':
                activity_data.update({
                    'icon': '🧘',
                    'details': [
                        f"{activity.name}",
                        f"{activity.metadata.get('description', 'Relaxation time')}",
                        f"Duration: {activity.duration_hours} hour(s)"
                    ]
                })
            
            day_data['activities'].append(activity_data)
        
        itinerary_data['days'].append(day_data)
    
    # Statistics
    poi_count = len(itinerary_state.get('selected_pois', []))
    restaurant_count = len(itinerary_state.get('selected_restaurants', []))
    
    poi_categories = {}
    for poi in itinerary_state.get('selected_pois', []):
        cat = poi.get('poi_category', 'General')
        poi_categories[cat] = poi_categories.get(cat, 0) + 1
    
    cuisines = {}
    for rest in itinerary_state.get('selected_restaurants', []):
        cuisine = str(rest.get('cuisine', 'Multi-cuisine'))[:30]
        cuisines[cuisine] = cuisines.get(cuisine, 0) + 1
    
    itinerary_data['statistics'] = {
        'poi_count': poi_count,
        'restaurant_count': restaurant_count,
        'poi_categories': poi_categories,
        'cuisines': dict(list(cuisines.items())[:3])
    }
    
# ===============================
# NORMALIZE RESTAURANT DATA
# ===============================
    normalized_restaurants = []
    for r in itinerary_state.get('selected_restaurants', []):
        normalized_restaurants.append({
        'name': r.get('restaurant_name') or r.get('name', 'Unknown'),
        'cuisine': r.get('cuisine', 'Multi-cuisine'),
        'avg_price': float(
            r.get('avg_price')
            or r.get('average_cost')
            or r.get('cost_for_two', 0)
        ),
        'min_price': float(r.get('min_price', 0)),
        'max_price': float(r.get('max_price', 0)),
        'rating': float(r.get('avg_rating') or r.get('rating', 0)),
        'sentiment': float(r.get('avg_sentiment') or r.get('sentiment', 0)),
        'location': r.get('location', 'Unknown'),
        'price_category': r.get('price_category', 'Unknown')
    })

# overwrite with normalized data
    itinerary_state['selected_restaurants'] = normalized_restaurants

    # Detailed Cost Breakdown
    hotel_cost = itinerary_state.get('selected_hotel', {}).get('total_cost', 0) if itinerary_state.get('selected_hotel') else 0
    poi_costs = sum(float(poi.get('Entry_fee_numeric', 0)) for poi in itinerary_state.get('selected_pois', []))
    restaurant_costs = sum(float(rest.get('avg_price', 0)) for rest in itinerary_state.get('selected_restaurants', []))
    total_cost = (
    itinerary_state.get('total_cost', 0)
    + restaurant_costs
)

    
    if total_cost > 0:
        hotel_percent = (hotel_cost / total_cost) * 100 if total_cost > 0 else 0
        poi_percent = (poi_costs / total_cost) * 100 if total_cost > 0 else 0
        restaurant_percent = (restaurant_costs / total_cost) * 100 if total_cost > 0 else 0
        
        transport_estimate = total_cost * 0.15
        misc_estimate = total_cost * 0.10
        grand_total = total_cost + transport_estimate + misc_estimate
        
        itinerary_data['cost_breakdown'] = {
            'hotel': {'cost': f"₹{hotel_cost:,.0f}", 'percent': f"{hotel_percent:.1f}%"},
            'poi': {'cost': f"₹{poi_costs:,.0f}", 'percent': f"{poi_percent:.1f}%"},
            'restaurant': {'cost': f"₹{restaurant_costs:,.0f}", 'percent': f"{restaurant_percent:.1f}%"},
            'transport': {'cost': f"₹{transport_estimate:,.0f}", 'percent': "15%"},
            'misc': {'cost': f"₹{misc_estimate:,.0f}", 'percent': "10%"},
            'grand_total': f"₹{grand_total:,.0f}"
        }
    else:
        itinerary_data['cost_breakdown'] = {
            'error': "No costs calculated (insufficient data)"
        }
    
    # Restaurants
    if itinerary_state.get('selected_restaurants'):
        for i, restaurant in enumerate(itinerary_state['selected_restaurants'], 1):
            itinerary_data['restaurants'].append({
                'index': i,
                'name': restaurant.get('name', 'Unknown'),
                'cuisine': restaurant.get('cuisine', 'Unknown'),
                'price_range': f"₹{restaurant.get('min_price', 0):,.0f} - ₹{restaurant.get('max_price', 0):,.0f}",
                'avg_price': f"₹{restaurant.get('avg_price', 0):,.0f}",
                'price_category': restaurant.get('price_category', 'Unknown'),
                'rating': f"{restaurant.get('rating', 0):.1f}",
                'sentiment': f"{restaurant.get('sentiment', 0):.2f}",
                'location': restaurant.get('location', 'Unknown')
            })
    
    # Tips based on trip type
    trip_type_tips = {
        'Family': [
            "👨‍👩‍👧‍👦 Look for family-friendly restaurants with kids' menus",
            "🕓 Dine early (5-6 PM) to avoid crowds with children",
            "📞 Make reservations for larger groups",
            "🥗 Check for buffet options - great value for families"
        ],
        'Couple': [
            "💑 Reserve romantic spots with ambiance",
            "🌇 Book evening slots for sunset views",
            "🍷 Consider wine pairing options",
            "🎵 Look for live music venues"
        ],
        'Solo': [
            "👤 Try bar seating for solo dining",
            "📱 Use apps for single-serving portions",
            "🗣️ Engage with locals for hidden gems",
            "🥡 Consider food tours or cooking classes"
        ],
        'Business': [
            "💼 Choose restaurants with private dining",
            "📶 Ensure good WiFi for meetings",
            "⏱️ Quick service options for lunch breaks",
            "🎤 Quiet ambiance for discussions"
        ],
        'Group': [
            "👥 Book in advance for large groups",
            "🍽️ Consider set menus for easier ordering",
            "💵 Split bills feature available at most restaurants",
            "🎉 Look for group discounts or packages"
        ],
        'Leisure': [
            "🧘 Try local specialties at your own pace",
            "📸 Instagram-worthy dining spots",
            "🌱 Explore different cuisines each day",
            "🛍️ Combine dining with local markets"
        ]
    }
    
    itinerary_data['tips'] = trip_type_tips.get(user_prefs.trip_type, [
        "🍽️ Make reservations during peak hours",
        "💵 Carry cash for smaller establishments",
        "🌶️ Ask about spice levels if sensitive",
        "⏰ Check opening hours - some restaurants close between meals"
    ])
    print("FOOD COST:", itinerary_data['cost_breakdown'].get('restaurant'))
    print("RESTAURANT COUNT:", len(itinerary_data['restaurants']))

    
    return itinerary_data

def generate_itinerary_thread(user_prefs_dict, session_id):
    """Thread function to generate itinerary in background"""
    try:
        # Create UserPreferences object
        user_prefs = UserPreferences(**user_prefs_dict)
        
        # Create RL environment
        env = RLItineraryEnvironment(user_prefs)
        
        # Create and train RL agent
        state_size = 2187
        max_actions = 20
        
        agent = QLearningAgent(
            state_size=state_size,
            action_size=max_actions,
            learning_rate=0.15,
            discount_factor=0.92,
            exploration_rate=0.5
        )
        
        # Train agent with fewer episodes for web
        agent.train(env, episodes=100)
        
        # Generate itinerary
        itinerary_state = agent.generate_itinerary(env)
        from collections import defaultdict

        enhanced = defaultdict(list)
        for day in range(user_prefs.days):
            enhanced[day] = itinerary_state['itinerary'].get(day, [])

        itinerary_state['itinerary'] = enhanced


        
        # Format itinerary for display
        itinerary_data = format_itinerary_for_html(itinerary_state, user_prefs)
        
        # Convert dataclasses to dictionaries for JSON serialization
        def convert_dataclass_to_dict(obj):
            if isinstance(obj, (list, tuple)):
                return [convert_dataclass_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_dataclass_to_dict(value) for key, value in obj.items()}
            elif hasattr(obj, '__dataclass_fields__'):
                return convert_dataclass_to_dict(asdict(obj))
            else:
                return obj
        
        # Store in user session - ensure everything is JSON serializable
        user_sessions[session_id] = {
            'status': 'completed',
            'itinerary': itinerary_data,
            'raw_itinerary': convert_dataclass_to_dict(itinerary_state),
            'user_prefs': convert_dataclass_to_dict(user_prefs)
        }
        
        print(f"✅ Itinerary generated for session {session_id}")
        
    except Exception as e:
        user_sessions[session_id] = {
            'status': 'error',
            'error': str(e)
        }
        print(f"❌ Error generating itinerary for session {session_id}: {e}")
        
def make_json_safe(obj):
    import pandas as pd

    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    return obj


# ========== FLASK ROUTES ==========

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    """Preferences page"""
    if request.method == 'POST':
        # Get form data
        user_prefs_dict = {
            'budget': float(request.form.get('budget', 20000)),
            'days': int(request.form.get('days', 3)),
            'trip_type': request.form.get('trip_type', 'Leisure'),
            'interests': request.form.getlist('interests'),
            'cuisine_preferences': request.form.getlist('cuisines'),
            'hotel_star_pref': float(request.form.get('hotel_stars', 3.0)),
            'meal_budget_per_day': float(request.form.get('meal_budget', 1000)),
            'transport_safety_weight': float(request.form.get('transport_safety', 0.7)),
            'adventure_level': request.form.get('adventure_level', 'moderate')
        }
        
        # Generate a unique session ID if not already exists
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        session_id = session['session_id']
        
        # Initialize session status
        user_sessions[session_id] = {
            'status': 'processing',
            'message': 'Generating your personalized itinerary...'
        }
        
        # Start itinerary generation in background thread
        thread = threading.Thread(target=generate_itinerary_thread, 
                                 args=(user_prefs_dict, session_id))
        thread.daemon = True
        thread.start()
        
        return redirect(url_for('loading'))
    
    # GET request - show preferences form
    return render_template('preferences.html')

@app.route('/loading')
def loading():
    """Loading page while itinerary is being generated"""
    return render_template('loading.html')

@app.route('/check_status')
def check_status():
    """Check itinerary generation status"""
    # Get session ID from session
    session_id = session.get('session_id')
    if session_id and session_id in user_sessions:
        return jsonify(make_json_safe(user_sessions[session_id]))
    return jsonify({'status': 'not_found'})

@app.route('/itinerary')
def itinerary():
    """Display generated itinerary"""
    # Get session ID from session
    session_id = session.get('session_id')
    if not session_id or session_id not in user_sessions:
        return redirect(url_for('preferences'))
    
    session_data = user_sessions[session_id]
    if session_data['status'] != 'completed':
        return redirect(url_for('loading'))
    
    return render_template('itinerary.html', 
                          itinerary=session_data['itinerary'])

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    """Chatbot interface"""
    # Get session ID from session
    session_id = session.get('session_id')
    if not session_id or session_id not in user_sessions:
        return redirect(url_for('preferences'))
    
    session_data = user_sessions[session_id]
    if session_data['status'] != 'completed':
        return redirect(url_for('loading'))
    
    if request.method == 'POST':
        # Handle chatbot query
        user_input = request.json.get('message', '')
        
        # Reconstruct UserPreferences from dictionary
        user_prefs_dict = session_data.get('user_prefs', {})
        user_prefs = UserPreferences(**user_prefs_dict)
        
        # Initialize chatbot with raw itinerary
        chatbot = TripChatbot(
            session_data.get('raw_itinerary', {}),
            user_prefs
        )
        
        # Generate response
        response = chatbot.generate_response(user_input)
        
        return jsonify({
            'response': response,
            'timestamp': datetime.now().strftime('%H:%M')
        })
    
    # GET request - show chatbot page
    return render_template('chatbot.html')

@app.route('/restart')
def restart():
    """Restart the planning process"""
    # Get session ID from session
    session_id = session.get('session_id')
    if session_id and session_id in user_sessions:
        del user_sessions[session_id]
    
    # Clear the session ID to start fresh
    session.pop('session_id', None)
    
    return redirect(url_for('preferences'))

@app.route('/export_itinerary')
def export_itinerary():
    """Export itinerary as text file"""
    # Get session ID from session
    session_id = session.get('session_id')
    if not session_id or session_id not in user_sessions:
        return "No itinerary found", 404
    
    session_data = user_sessions[session_id]
    if session_data['status'] != 'completed':
        return "Itinerary not ready", 400
    
    # Generate text version of itinerary
    itinerary_text = generate_text_itinerary(session_data['itinerary'])
    
    # Create response with text file
    from flask import make_response
    response = make_response(itinerary_text)
    response.headers["Content-Disposition"] = "attachment; filename=my_itinerary.txt"
    response.headers["Content-type"] = "text/plain"
    return response

def generate_text_itinerary(itinerary_data):
    """Generate text version of itinerary"""
    text = "=" * 80 + "\n"
    text += "🌟 AI-GENERATED PERSONALIZED ITINERARY 🌟\n"
    text += "=" * 80 + "\n\n"
    
    # Trip Summary
    text += "📋 TRIP SUMMARY\n"
    text += "-" * 60 + "\n"
    text += f"Trip Type: {itinerary_data['trip_summary']['trip_type']}\n"
    text += f"Duration: {itinerary_data['trip_summary']['days']} days\n"
    text += f"Total Budget: {itinerary_data['trip_summary']['budget']}\n"
    text += f"Estimated Cost: {itinerary_data['trip_summary']['estimated_cost']}\n"
    text += f"Remaining Budget: {itinerary_data['trip_summary']['remaining_budget']}\n"
    text += f"Budget Utilization: {itinerary_data['trip_summary']['utilization']}\n\n"
    
    # Hotel
    if itinerary_data['hotel']['name']:
        text += "🏨 SELECTED HOTEL\n"
        text += "-" * 60 + "\n"
        text += f"Hotel: {itinerary_data['hotel']['name']}\n"
        text += f"Star Rating: {itinerary_data['hotel']['stars']}\n"
        text += f"Price per night: {itinerary_data['hotel']['price_per_night']}\n"
        text += f"Total cost: {itinerary_data['hotel']['total_cost']}\n\n"
    
    # Day-wise itinerary
    text += "📅 DAY-WISE ITINERARY\n"
    text += "-" * 60 + "\n"
    for day in itinerary_data['days']:
        text += f"\nDay {day['day_number']}:\n"
        for activity in day['activities']:
            text += f"  {activity['time']}: {activity['name']}\n"
    
    return text

# Run the app
if __name__ == '__main__':
    # Check if datasets are loaded
    if food_df is None or hotel_df is None:
        print("❌ Datasets not loaded properly. Check your CSV files.")
    else:
        print("✅ Flask app ready to run!")
    
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)