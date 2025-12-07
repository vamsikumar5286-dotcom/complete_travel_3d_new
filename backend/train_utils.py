"""
train_utils.py
Shared utilities for train pricing logic.
"""
import pandas as pd
import numpy as np
import re


class TrainPriceLookup:
    """Handles Train pricing lookup from part_1 and part_2 datasets."""
    
    def __init__(self, part_1_df=None, part_2_df=None):
        self.base_map = {}
        self.distance_df = None
        
        if part_1_df is not None:
            self._load_base_prices(part_1_df)
        if part_2_df is not None:
            self._load_distance_prices(part_2_df)
    
    def _load_base_prices(self, df):
        """Load base prices from part_1.csv using the 'Base price' column."""
        try:
            df = df.copy()
            # Detect base price column
            cols_lower = {c.lower(): c for c in df.columns}
            
            if 'base price' in cols_lower:
                base_col = cols_lower['base price']
                key_col = df.columns[0]  # First column is train type
                self.base_map = dict(zip(
                    df[key_col].astype(str).str.strip(), 
                    pd.to_numeric(df[base_col], errors='coerce').fillna(0.0).astype(float)
                ))
                print(f"Loaded base prices for {len(self.base_map)} train types from part_1.csv")
                return

            print("Warning: Could not detect base price column in part_1.csv")
        except Exception as e:
            print(f"Warning: Could not load base prices: {e}")
    
    def _load_distance_prices(self, df):
        """Load distance price lookup table from train_dataset_part2.csv."""
        try:
            self.distance_df = df.copy()
            print(f"Loaded distance price table with {len(self.distance_df)} rows from train_dataset_part2.csv")
        except Exception as e:
            print(f"Warning: Could not load distance prices: {e}")
    
    def get_base_price(self, train_type):
        """Get base price for a train type."""
        if not train_type:
            return 0.0
        # Friendly name mapping -> canonical CSV column keys
        name_mapping = {
            'ordinary passenger': 'Ord_Passenger',
            'ordinary': 'Ord_Passenger',
            'ordinary local': 'Ord_LocalSuburb',
            'ordinary local suburb': 'Ord_LocalSuburb',
            'ordinary sleeper': 'Ord_SL',
            'ordinary first class': 'Ord_FC',
            'express 2s': 'Exp_2S',
            'express sleeper': 'Exp_SL',
            'express first class': 'Exp_FC',
            'ac chair car': 'AC_CC',
            'ac 3 tier': 'AC_3A',
            'ac 3a': 'AC_3A',
            'ac 2 tier': 'AC_2A',
            'ac 2a': 'AC_2A',
            'ac 1 tier': 'AC_1A',
            'ac 1a': 'AC_1A',
            'garibrath chair car': 'GR_CC',
            'garibrath 3 tier': 'GR_3A',
            'rajdhani 3 tier': 'Raj_3A',
            'rajdhani 2 tier': 'Raj_2A',
            'rajdhani 1 tier': 'Raj_1A',
            'shatabdi chair car': 'Sha_CC',
            'shatabdi executive': 'Sha_EC',
            'jan shatabdi 2s': 'JS_2S',
            'jan shatabdi chair car': 'JS_CC',
            'yuva': 'Yuva_Other',
            'yuva 18-35': 'Yuva_18_35',
        }

        key = str(train_type).strip()
        key_lower = key.lower()

        # Check mapping first
        mapped = name_mapping.get(key_lower)
        if mapped and mapped in self.base_map:
            return float(self.base_map[mapped])

        # Exact match
        if key in self.base_map:
            return float(self.base_map[key])

        # Fallback to lower-case matching
        lowered = {k.lower(): v for k, v in self.base_map.items()}
        return float(lowered.get(key_lower, 0.0))
    
    def get_distance_price(self, train_type, distance_km):
        """Get distance price by matching distance tier and train class column.
        
        Uses canonical mapping and picks the closest distance tier.
        """
        if self.distance_df is None or distance_km is None:
            return 0.0
        
        try:
            df = self.distance_df.copy()
            
            # Parse distance tiers
            if 'Distance' in df.columns:
                df['dist_mid'] = df['Distance'].apply(self._parse_distance_tier)
            else:
                return 0.0
            
            df = df.dropna(subset=['dist_mid'])
            if df.empty:
                return 0.0

            # Map canonical train types to column names
            # Map many user-friendly variants to the correct columns
            class_to_cols = {
                'ordinary': ['Ord_LocalSuburb', 'Ord_Passenger', 'Ord_SL', 'Ord_FC'],
                'ordinary passenger': ['Ord_Passenger', 'Ord_LocalSuburb', 'Ord_SL', 'Ord_FC'],
                'ordinary local': ['Ord_LocalSuburb', 'Ord_Passenger', 'Ord_SL', 'Ord_FC'],
                'express': ['Exp_2S', 'Exp_SL', 'Exp_FC'],
                'express 2s': ['Exp_2S', 'Exp_SL', 'Exp_FC'],
                'ac': ['AC_CC', 'AC_3A', 'AC_2A', 'AC_1A'],
                'ac 3a': ['AC_3A', 'AC_2A', 'AC_1A', 'AC_CC'],
                'garibrath': ['GR_CC', 'GR_3A'],
                'rajdhani': ['Raj_3A', 'Raj_2A', 'Raj_1A'],
                'shatabdi': ['Sha_CC', 'Sha_EC'],
                'jan-shatabdi': ['JS_2S', 'JS_CC'],
                'jan shatabdi': ['JS_2S', 'JS_CC'],
                'yuva': ['Yuva_Other', 'Yuva_18_35'],
                'yuva-cc': ['Yuva_Other', 'Yuva_18_35']
            }
            
            # Normalize train_type: accept canonical keys (e.g. 'Ord_Passenger') or friendly names
            train_type_str = str(train_type).strip() if train_type else ''
            train_type_lower = train_type_str.lower()

            # If user passed a canonical column name that exists in the distance_df, use it directly
            canonical_candidates = []
            if train_type_str in df.columns:
                canonical_candidates = [train_type_str]

            # If not canonical, try friendly mapping keys
            candidates = canonical_candidates or class_to_cols.get(train_type_lower, [])

            # Also allow matching against base_map keys (case-insensitive)
            if not candidates and hasattr(self, 'base_map') and self.base_map:
                lowered_base = {k.lower(): k for k in self.base_map.keys()}
                if train_type_lower in lowered_base:
                    # use the canonical base_map key as the column name
                    candidates = [lowered_base[train_type_lower]]

            if not candidates:
                return 0.0

            # Find closest distance tier
            distance_float = float(distance_km)
            df['delta'] = (df['dist_mid'] - distance_float).abs()
            
            # Use the closest tier
            closest_row = df.loc[df['delta'].idxmin()]
            
            # Return first valid numeric value from candidates
            for col in candidates:
                if col in closest_row.index:
                    try:
                        val = pd.to_numeric(closest_row[col], errors='coerce')
                        if not pd.isna(val) and float(val) > 0:
                            return float(val)
                    except Exception:
                        pass
            
            return 0.0
        except Exception as e:
            print(f"Warning: Could not get distance price: {e}")
            return 0.0
    
    @staticmethod
    def _parse_distance_tier(s):
        """Parse distance tier strings like '01-05' or '16-20' into midpoint."""
        try:
            if pd.isna(s):
                return None
            s = str(s).strip()
            # Extract all numbers using regex
            nums = re.findall(r"\d+", s)
            if not nums:
                return None
            nums = [int(n) for n in nums]
            if len(nums) == 1:
                return float(nums[0])
            # Two numbers: use average as midpoint
            a, b = nums[0], nums[1]
            return float((a + b) / 2.0)
        except Exception:
            return None

