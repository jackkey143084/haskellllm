-- | BF16 (bfloat16) support.
--
-- We store bf16 as a raw 'Data.Word.Word16' and convert on the fly, exactly
-- like the Go port: bf16 is the truncated top half of an IEEE-754 float32.
--
-- Decode:  word16 -> word32 (shifted left 16 bits) -> reinterpret as Float.
-- Encode:  float32 -> word32 -> round-to-nearest-even -> top 16 bits.
module HQ.BF16
  ( BF16(..)
  , bf16ToF32
  , f32ToBF16
  , bf16ListToF32
  , f32ListToBF16
  ) where

import Data.Bits (shiftL, shiftR, (.&.), (.|.))
import Data.Word (Word16, Word32)
import GHC.Float (castWord32ToFloat, castFloatToWord32)

-- | A raw bfloat16 value, stored as the upper 16 bits of a float32.
newtype BF16 = BF16 Word16
  deriving (Eq, Ord, Show)

-- | Decode bf16 to float32. Exact: bf16 IS a truncated float32.
bf16ToF32 :: BF16 -> Float
bf16ToF32 (BF16 w) = castWord32ToFloat (fromIntegral w `shiftL` 16)
{-# INLINE bf16ToF32 #-}

-- | Encode float32 to bf16 with round-to-nearest-even.
--   NaN payloads are truncated (top bits kept), matching ML convention.
f32ToBF16 :: Float -> BF16
f32ToBF16 x =
  let u = castFloatToWord32 x
      lsb = (u `shiftR` 16) .&. 1           -- ties-to-even guard bit
      bias = 0x7FFF + lsb
      rounded = (u + bias) `shiftR` 16
  in BF16 (fromIntegral rounded)
{-# INLINE f32ToBF16 #-}

bf16ListToF32 :: [BF16] -> [Float]
bf16ListToF32 = map bf16ToF32
{-# INLINE bf16ListToF32 #-}

f32ListToBF16 :: [Float] -> [BF16]
f32ListToBF16 = map f32ToBF16
{-# INLINE f32ListToBF16 #-}
