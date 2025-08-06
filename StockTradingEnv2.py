# Import only necessary modules to avoid circular imports
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from parameters import INITIAL_ACCOUNT_BALANCE, COST_PER_TRADE, BUYTHRESHOLD, SELLTHRESHOLD

class StockTradingEnv2(gym.Env):
    """A stock trading environment for OpenAI gym""" 
    
    # Class variables for tracking actions across all instances
    action_dict = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    amount_dict = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
    
    def __init__(self, df,NLAGS = 5,NUMVARS = 4,MAXIMUM_SHORT_VALUE = INITIAL_ACCOUNT_BALANCE,INITIAL_ACCOUNT_BALANCE=INITIAL_ACCOUNT_BALANCE,MAX_STEPS=20000,finalsignalsp=[],INITIAL_NET_WORTH=INITIAL_ACCOUNT_BALANCE, INITIAL_SHARES_HELD=0,COST_PER_TRADE=COST_PER_TRADE,BUYTHRESHOLD=BUYTHRESHOLD,SELLTHRESHOLD=SELLTHRESHOLD):
        super(StockTradingEnv2, self).__init__()
        self.df = df
        self.NLAGS = NLAGS
        self.NUMVARS = NUMVARS
        self.MAXIMUM_SHORT_VALUE = MAXIMUM_SHORT_VALUE
        self.INITIAL_ACCOUNT_BALANCE = INITIAL_ACCOUNT_BALANCE
        self.INITIAL_NET_WORTH=INITIAL_NET_WORTH
        self.INITIAL_SHARES_HELD=INITIAL_SHARES_HELD
        self.MAX_STEPS = MAX_STEPS
        self.finalsignalsp = finalsignalsp
        self.COST_PER_TRADE = COST_PER_TRADE
        self.BUYTHRESHOLD = BUYTHRESHOLD
        self.SELLTHRESHOLD = SELLTHRESHOLD
        
        # Track current date for daily position liquidation
        self.current_date = None
        self.previous_date = None
        
        # Track daily performance for end-of-day rewards
        self.daily_start_net_worth = INITIAL_NET_WORTH
        self.daily_liquidation_reward = 0.0

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.float32(np.array([-1, 0])), high=np.float32(np.array([1, 1])), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.float32(0), high=np.float32(1), shape=(self.NUMVARS, self.NLAGS+1), dtype=np.float32)

    def _next_observation(self):
        # Get the stock data points for the last NLAGS periods
        data_slices = []
        for sig in self.finalsignalsp:
            data_slices.append(self.df.loc[(self.current_step-self.NLAGS): self.current_step, sig])
        
        if data_slices:
            sigframe = pd.concat(data_slices, axis=1, keys=self.finalsignalsp)
        else:
            sigframe = pd.DataFrame()
        
        obs = sigframe.T.to_numpy().astype(np.float32)
        
        return obs

    def _liquidate_daily_positions(self):
        """Optimized daily position liquidation for GPU training speed"""
        if self.shares_held == 0:
            # Fast path for no positions - just calculate reward
            daily_pnl = self.net_worth - self.daily_start_net_worth
            self._calculate_daily_liquidation_reward(daily_pnl, 0)
            return
        
        # Efficient liquidation calculation
        current_price = self.df.loc[self.current_step, "vwap2"]
        shares_abs = abs(self.shares_held)
        gross_value = shares_abs * current_price
        transaction_cost = gross_value * self.COST_PER_TRADE
        
        if self.shares_held > 0:
            # Long liquidation: sell shares
            net_proceeds = gross_value - transaction_cost
            self.balance += net_proceeds
            liquidation_pnl = -transaction_cost  # Only transaction cost as loss
        else:
            # Short liquidation: cover position
            total_cost = gross_value + transaction_cost
            self.balance -= total_cost
            liquidation_pnl = -transaction_cost  # Only transaction cost as loss
        
        # Reset position and update net worth in one step
        self.shares_held = 0
        self.net_worth = self.balance  # shares_held is now 0
        
        # Calculate rewards and reset daily tracking
        total_daily_pnl = self.net_worth - self.daily_start_net_worth
        self._calculate_daily_liquidation_reward(total_daily_pnl, liquidation_pnl)
        self.daily_start_net_worth = self.net_worth
    
    def _calculate_daily_liquidation_reward(self, total_daily_pnl, liquidation_pnl):
        """Optimized daily reward calculation for GPU training speed"""
        daily_return_pct = total_daily_pnl / self.INITIAL_ACCOUNT_BALANCE
        
        # Simplified reward calculation - remove expensive computations
        base_reward = daily_return_pct * (500 if daily_return_pct > 0 else 200)
        
        # Simple liquidation penalty (transaction costs)
        liquidation_penalty = liquidation_pnl / self.INITIAL_ACCOUNT_BALANCE * 100
        
        # Minimal consistency tracking - only keep last 5 returns
        if not hasattr(self, 'daily_returns_history'):
            self.daily_returns_history = []
        
        self.daily_returns_history.append(daily_return_pct)
        if len(self.daily_returns_history) > 5:
            self.daily_returns_history.pop(0)  # Remove oldest, keep last 5
        
        # Simple consistency bonus without expensive std calculation
        if len(self.daily_returns_history) >= 3:
            avg_return = sum(self.daily_returns_history) / len(self.daily_returns_history)
            consistency_bonus = avg_return * 25  # Simplified bonus
        else:
            consistency_bonus = 0
        
        self.daily_liquidation_reward = base_reward + liquidation_penalty + consistency_bonus

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, "vwap2"]
        prev_net_worth = self.net_worth
        
        action_type = action[0]
        amount = max(0.1, min(1.0, action[1]))  # Clamp amount between 0.1 and 1.0
        
        # Signal change threshold - only trade if signals have changed significantly
        # signal_change_threshold = 0.25
        # 
        # if hasattr(self, 'previous_obs') and self.previous_obs is not None and self.current_step > self.NLAGS + 5:
        #     # Calculate signal change from recent observations
        #     current_obs = self._next_observation()
        #     signal_change = np.mean(np.abs(current_obs.flatten() - self.previous_obs.flatten()))
        #     if signal_change < signal_change_threshold and abs(action_type) > BUYTHRESHOLD:
        #         action_type = 0  # Force hold if signals haven't changed significantly
        
        # Calculate transaction costs (0.1% per trade)
        transaction_cost_rate = self.COST_PER_TRADE
        
        if action_type >= BUYTHRESHOLD:  
          max_affordable = self.balance / current_price
          shares_to_buy = int(max_affordable * amount)
          
          if shares_to_buy > 0:
              cost = shares_to_buy * current_price
              transaction_cost = cost * transaction_cost_rate
              total_cost = cost + transaction_cost
              
              if total_cost <= self.balance:
                  self.balance -= total_cost
                  self.shares_held += shares_to_buy
                  StockTradingEnv2.action_dict['BUY'] += 1
                  StockTradingEnv2.amount_dict['BUY'] += amount
                  
        elif action_type <= SELLTHRESHOLD:  # Sell threshold
          max_sellable = self.shares_held + (self.MAXIMUM_SHORT_VALUE / current_price)
          shares_to_sell = int(max_sellable * amount)
          
          if shares_to_sell > 0:
              revenue = shares_to_sell * current_price
              transaction_cost = revenue * transaction_cost_rate
              net_revenue = revenue - transaction_cost
              
              self.balance += net_revenue
              self.shares_held -= shares_to_sell
              StockTradingEnv2.action_dict['SELL'] += 1
              StockTradingEnv2.amount_dict['SELL'] += amount
        else:
          StockTradingEnv2.action_dict['HOLD'] += 1
          StockTradingEnv2.amount_dict['HOLD'] += amount
          
        # Update net worth
        self.net_worth = self.balance + self.shares_held * current_price
        
        # Store action info for reward calculation
        self.prev_net_worth = prev_net_worth
        self.action_taken = action_type
        
        # Track action history for over-activity penalty
        if not hasattr(self, 'action_history'):
            self.action_history = []
        self.action_history.append(abs(action_type))
        if len(self.action_history) > 20:  # Keep last 20 actions
            self.action_history = self.action_history[-20:] 

    def step(self, action):
        noisy = len(action) == 1
        if noisy:
            action = action[0]

        # Check for date change and liquidate positions if needed
        daily_liquidation_occurred = False
        if 'currentdate' in self.df.columns:
            current_date = self.df.loc[self.current_step, 'currentdate']
            if self.current_date is not None and current_date != self.current_date:
                # New day started, liquidate all positions from previous day
                self._liquidate_daily_positions()
                daily_liquidation_occurred = True
            self.previous_date = self.current_date
            self.current_date = current_date

        current_price = self.df.loc[self.current_step, "vwap2"]
        prev_price = self.df.loc[self.current_step-1, "vwap2"] if self.current_step > 0 else current_price
        
        self._take_action(action)
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        profit_pct = profit / INITIAL_ACCOUNT_BALANCE
        
        # Reward components
        profit_reward = profit_pct * 100
        step_pnl = self.net_worth - self.prev_net_worth
        step_reward = step_pnl / INITIAL_ACCOUNT_BALANCE * 1000
        price_change = (current_price - prev_price) / prev_price
        
        # Optimized market trend calculation - avoid list comprehension
        if self.current_step >= 10:
            start_idx = max(0, self.current_step-10)
            start_price = self.df.loc[start_idx, "vwap2"]
            market_trend = (current_price - start_price) / start_price
        else:
            market_trend = 0
        
        action_reward = 0
        position_reward = 0
        
        if hasattr(self, 'action_taken'):
            if self.action_taken > BUYTHRESHOLD:  # Buy action
                if price_change > 0:
                    action_reward = 5
                elif market_trend > 0.002:
                    action_reward = 3
                else:
                    action_reward = -1
            elif self.action_taken < SELLTHRESHOLD:  # Sell action
                if price_change < 0:
                    action_reward = 5
                elif market_trend < -0.002:
                    action_reward = 3
                else:
                    action_reward = -1
            else:  # Hold action
                if abs(price_change) < 0.002:
                    action_reward = 2
                else:
                    action_reward = 0
            
            # Balanced position holding rewards - equal treatment for long/short
            if self.shares_held < 0 and market_trend < -0.002:  # Short position in downtrend
                position_reward = abs(self.shares_held) * current_price * abs(market_trend) * 15  # Reduced from 30
                if hasattr(self, 'position_hold_time'):
                    self.position_hold_time += 1
                    if self.position_hold_time > 5:  # Reduced from 8
                        position_reward += 1  # Reduced from 2
                else:
                    self.position_hold_time = 1
            elif self.shares_held > 0 and market_trend > 0.002:  # Long position in uptrend
                position_reward = self.shares_held * current_price * abs(market_trend) * 15  # Now uses abs() like shorts
                if hasattr(self, 'position_hold_time'):
                    self.position_hold_time += 1
                    if self.position_hold_time > 5:  # Reduced from 8
                        position_reward += 1  # Reduced from 2
                else:
                    self.position_hold_time = 1
            else:
                self.position_hold_time = 0
        
        # Risk penalty
        portfolio_value = abs(self.shares_held * current_price)
        leverage_ratio = portfolio_value / INITIAL_ACCOUNT_BALANCE
        risk_penalty = -max(0, (leverage_ratio - 2.0) * 10)
        # Simplified activity penalty - reduce computational overhead
        activity_penalty = 0
        if hasattr(self, 'action_history') and len(self.action_history) >= 10:
            # Simple recent activity check without expensive operations
            recent_actions = self.action_history[-10:]
            recent_activity = sum(abs(a) for a in recent_actions) / len(recent_actions)
            if recent_activity > 0.25:
                activity_penalty = -(recent_activity - 0.25) * 60
            
            # Count consecutive high-activity actions more efficiently
            consecutive_trades = 0
            for action in reversed(self.action_history[-5:]):
                if abs(action) > BUYTHRESHOLD:
                    consecutive_trades += 1
                else:
                    break
            if consecutive_trades >= 2:
                activity_penalty -= consecutive_trades * 10
        
        # Optimized Sharpe-like reward - reduce memory allocations
        if not hasattr(self, 'returns_history'):
            self.returns_history = []
        
        step_return = step_pnl / INITIAL_ACCOUNT_BALANCE
        self.returns_history.append(step_return)
        if len(self.returns_history) > 20:  # Reduced from 50 for speed
            self.returns_history.pop(0)  # Remove oldest
        
        # Simplified Sharpe calculation
        sharpe_reward = 0
        if len(self.returns_history) >= 10:
            avg_return = sum(self.returns_history) / len(self.returns_history)
            # Approximate std with simpler calculation
            variance_sum = sum((r - avg_return) ** 2 for r in self.returns_history)
            std_return = (variance_sum / len(self.returns_history)) ** 0.5 + 1e-8
            sharpe_reward = (avg_return / std_return) * 10
        
        # Simplified balance reward - reduce computational overhead
        balance_reward = 0
        if hasattr(self, 'action_history') and len(self.action_history) >= 20:
            # Simplified balance check - just ensure some variety exists
            recent_actions = self.action_history[-20:]
            has_buy = any(a > BUYTHRESHOLD for a in recent_actions)
            has_sell = any(a < SELLTHRESHOLD for a in recent_actions)
            has_hold = any(abs(a) <= BUYTHRESHOLD for a in recent_actions)
            
            # Simple bonus for having variety in actions
            variety_count = sum([has_buy, has_sell, has_hold])
            balance_reward = (variety_count - 1) * 1.5  # Reward for 2-3 different action types
        
        # Reward scaling functions
        def scale_reward_component(value, scale_factor=1.0):
            if abs(value) <= 1e-8:
                return 0.0
            scaled = np.tanh(value * scale_factor)
            return np.clip(scaled, -1.0, 1.0)
        
        def scale_penalty_component(value, scale_factor=1.0):
            if value >= 0:
                return 0.0
            scaled = np.tanh(value * scale_factor)
            return np.clip(scaled, -1.0, 1.0)
        
        # Add daily liquidation reward component
        daily_liquidation_component = 0.0
        if daily_liquidation_occurred and hasattr(self, 'daily_liquidation_reward'):
            daily_liquidation_component = self.daily_liquidation_reward
            # Reset the daily liquidation reward after using it
            self.daily_liquidation_reward = 0.0
        
        # Scale and combine reward components
        scaled_profit = scale_reward_component(profit_reward, scale_factor=0.2)  # Increased from 0.1
        scaled_step = scale_reward_component(step_reward, scale_factor=0.1)  # Reduced from 0.5
        scaled_action = scale_reward_component(action_reward, scale_factor=0.1)
        scaled_position = scale_reward_component(position_reward, scale_factor=0.15)
        scaled_risk = scale_penalty_component(risk_penalty, scale_factor=0.05)  # Slightly increased
        scaled_activity = scale_penalty_component(activity_penalty, scale_factor=0.15)
        scaled_sharpe = scale_reward_component(sharpe_reward, scale_factor=0.15)  # Slightly increased
        scaled_balance = scale_reward_component(balance_reward, scale_factor=0.05)
        scaled_daily_liquidation = scale_reward_component(daily_liquidation_component, scale_factor=0.05)
        
        reward = (scaled_profit * 0.3 + scaled_step * 0.1 + scaled_action * 0.1 +
                 scaled_position * 0.05 + scaled_risk * 0.05 + scaled_activity * 0.05 +
                 scaled_sharpe * 0.25 + scaled_balance * 0.05 + scaled_daily_liquidation * 0.05)
        
        #Scale to [-0.1, 0.1] range and ensure bounds
        reward = float(np.clip(reward * 0.1, -0.1, 0.1))
        
        self.current_step += 1
        
        # Episode termination conditions
        done = (self.net_worth <= INITIAL_ACCOUNT_BALANCE * 0.5 or 
                self.current_step >= min(self.MAX_STEPS, self.df.shape[0]-1) or 
                self.net_worth <= 0)
        
        obs = self._next_observation()
        
        # Update previous observation for signal change tracking
        self.previous_obs = obs.copy()
        
        info = {
            'profit': profit,
            'profit_pct': profit_pct,
            'step_pnl': step_pnl,
            'net_worth': self.net_worth,
            'shares_held': self.shares_held,
            'balance': self.balance,
            'action_reward': action_reward,
            'position_reward': position_reward,
            'risk_penalty': risk_penalty,
            'activity_penalty': activity_penalty,
            'sharpe_reward': sharpe_reward,
            'balance_reward': balance_reward,
            'daily_liquidation_reward': daily_liquidation_component,
            'market_trend': market_trend if 'market_trend' in locals() else 0,
            'recent_activity': np.mean(self.action_history[-10:]) if hasattr(self, 'action_history') and len(self.action_history) >= 10 else 0
        }
        
        # Episode reward and length when done
        if done:
            episode_profit_pct = profit / INITIAL_ACCOUNT_BALANCE
            scaled_episode_reward = np.clip(np.tanh(episode_profit_pct * 5) * 1, -1, 1)
            info['r'] = scaled_episode_reward
            info['l'] = self.current_step
        
                  
        # Gymnasium format: observation, reward, terminated, truncated, info
        terminated = done and self.current_step >= len(self.df) - 1  # Episode ended naturally
        truncated = done and not terminated  # Episode ended due to other conditions
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        if seed is not None:
            np.random.seed(seed)
            
        self.balance = self.INITIAL_ACCOUNT_BALANCE
        self.net_worth = self.INITIAL_NET_WORTH
        self.shares_held = self.INITIAL_SHARES_HELD
        self.current_step = self.NLAGS
        
        # Initialize date tracking
        if 'currentdate' in self.df.columns:
            self.current_date = self.df.loc[self.current_step, 'currentdate']
            self.previous_date = None
        else:
            self.current_date = None
            self.previous_date = None
        
        # Reset tracking variables
        self.prev_net_worth = self.INITIAL_ACCOUNT_BALANCE
        self.action_taken = 0
        self.returns_history = []
        # Initialize action history with balanced actions to prevent initial bias
        self.action_history = [self.BUYTHRESHOLD+0.1, self.SELLTHRESHOLD-0.1, 0.0, self.BUYTHRESHOLD+0.1, self.SELLTHRESHOLD-0.1, 0.0] * 3  # Mix of buy, sell, hold
        self.market_direction_history = []
        self.position_hold_time = 0
        self.previous_obs = None
        self.previous_signals = None
        
        # Reset daily tracking variables
        self.daily_start_net_worth = self.INITIAL_NET_WORTH
        self.daily_liquidation_reward = 0.0
        self.daily_returns_history = []
        
        observation = self._next_observation()
        info = {}
        return observation, info
    
    def seed(self, seed=None):
        np.random.seed(seed)
        return [seed]
