#!/usr/bin/env python
"""Test script to simulate occupancy changes and verify appliance state updates."""

import sys
from firebase_helper import publish_room_state, firebase_is_configured

def main():
    if not firebase_is_configured():
        print("❌ Firebase is not configured. Please set up your credentials and database URL.")
        print("   Check your .env file or environment variables.")
        return

    if len(sys.argv) > 1:
        try:
            occupancy = int(sys.argv[1])
        except ValueError:
            print(f"❌ Invalid occupancy count: '{sys.argv[1]}'. Please provide a number.")
            return
    else:
        try:
            occupancy = int(input("Enter occupancy count (0-10): "))
        except ValueError:
            print("❌ Invalid input. Please enter a number.")
            return

    if occupancy < 0:
        print("❌ Occupancy count cannot be negative.")
        return

    print(f"\n📡 Publishing occupancy count: {occupancy}")
    print("-" * 50)

    try:
        result = publish_room_state(occupancy)
        
        print(f"✅ Successfully published to Firebase!")
        print(f"\n📊 Room State:")
        print(f"   Occupancy: {result['occupancy']}")
        print(f"   Lights: {result['lightsStatus']} (≥1 occupant)")
        print(f"   Fans: {result['fansStatus']} (≥2 occupants)")
        print(f"   AC: {result['airconStatus']} (≥3 occupants)")
        print(f"   Updated at: {result['updatedAt']}")
        print("-" * 50)
        
    except Exception as e:
        print(f"❌ Error publishing to Firebase: {e}")

if __name__ == "__main__":
    main()
