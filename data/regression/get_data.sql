SELECT 
    p.z AS specz,           -- Y: The target variable (Spectroscopic Redshift)
    p.dered_u,              -- X: Dereddened magnitude in the u-band
    p.dered_g,              -- X: Dereddened magnitude in the g-band
    p.dered_r,              -- X: Dereddened magnitude in the r-band
    p.dered_i,              -- X: Dereddened magnitude in the i-band
    p.dered_z,              -- X: Dereddened magnitude in the z-band
    p.zErr                  -- (Optional) Spectroscopic Redshift Error
FROM 
    SpecPhoto p             -- Use the combined SpecPhoto view
WHERE 
    p.class = 'GALAXY'      -- Only select objects classified as galaxies
    AND p.z > 0.001         -- Exclude very low or questionable redshifts
    AND p.zWarning = 0      -- Ensure the spectroscopic measurement is clean
    AND p.dered_r < 19.0