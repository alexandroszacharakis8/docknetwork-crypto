/*!
This file is derived from `sw_scalar_mult.rs` in the original project.
Changes: this variant removes commitments to scalars and relevant values; the rest of the code is otherwise unchanged.
SPDX-License-Identifier: Apache-2.0

Proof of scalar multiplication on short Weierstrass curve without committing to the scalar, but simply proven knowledge of it.
This is essentially the same as [sw_scalar_mult](crate::ec::sw_scalar_mult)

The protocol proves that for committed curve point `S` and public curve point `R`, `S = R * omega` where omega is a witness.
The verifier only has commitments to `S`'s coordinates `x` and `y`.

The idea is the prover generates a random point say `J = R * alpha` and the point `K` such that `K = (alpha - omega) * R`
Now it proves using the protocol of point addition that sum of points `S` and `K` is point `J` and it knows
the opening of these points. `alpha` is chosen to not be either of `(0, omega, 2*omega)` to avoid point doubling or points
at infinity in point addition protocol. The prover repeats this protocol several times as per the security parameter of the protocol

*/

use crate::{
    ec::{
        commitments::{
            point_coords_as_scalar_field_elements, PointCommitment, PointCommitmentWithOpening,
            SWPoint,
        },
        sw_point_addition::{PointAdditionProof, PointAdditionProtocol},
    },
    error::Error,
};
use ark_ec::{short_weierstrass::Affine, AffineRepr, CurveGroup};
use ark_ff::{Field, One, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{cfg_into_iter, cfg_iter, io::Write, ops::Neg, rand::RngCore, vec::Vec, UniformRand};
use dock_crypto_utils::{
    commitment::PedersenCommitmentKey, msm::WindowTable,
    randomized_mult_checker::RandomizedMultChecker,
};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Protocol for proving scalar multiplication with committed point and witnessed scalar.
/// `P` is the curve where the points live and `C` is the curve where commitments (to their coordinates) live.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ScalarMultiplicationWCProtocol<P: SWPoint, C: SWPoint, const NUM_REPS: usize = 128> {
    /// The witnessed scalar
    pub omega: P::ScalarField,
    sub_protocols: Vec<ScalarMultiplicationWCProtocolSingleRep<P, C>>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ScalarMultiplicationWCProtocolSingleRep<P: SWPoint, C: SWPoint> {
    /// random dlog `alpha`
    pub alpha: P::ScalarField,
    /// Commitment to the point `alpha * R`
    pub comm_alpha_point: PointCommitmentWithOpening<C>,
    /// Commitment to the point `(alpha - omega) * R`
    pub comm_alpha_minus_omega_point: PointCommitmentWithOpening<C>,
    pub add: PointAdditionProtocol<P, C>,
}

/// Proof of scalar multiplication with committed point.
/// `P` is the curve where the points live and `C` is the curve where commitments (to their coordinates) live.
#[derive(Clone, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ScalarMultiplicationWCProof<P: SWPoint, C: SWPoint, const NUM_REPS: usize = 128>(
    Vec<ScalarMultiplicationWCProofSingleRep<P, C>>,
);

#[derive(Clone, PartialEq, Eq, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct ScalarMultiplicationWCProofSingleRep<P: SWPoint, C: SWPoint> {
    /// Commitment to the point `alpha * R`
    pub comm_alpha_point: PointCommitment<C>,
    /// Commitment to the point `(alpha - omega) * R`
    pub comm_alpha_minus_omega_point: PointCommitment<C>,
    pub add: PointAdditionProof<P, C>,
    /// z1 holds alpha or omega-alpha
    pub z1: P::ScalarField,
    pub z3: C::ScalarField,
    pub z4: C::ScalarField,
}

impl<P: SWPoint, C: SWPoint, const NUM_REPS: usize> ScalarMultiplicationWCProtocol<P, C, NUM_REPS> {
    /// For proving `base * scalar = result` where  `comm_result` is a commitment to  `result` and scalar is a witness
    pub fn init<R: RngCore>(
        rng: &mut R,
        omega: P::ScalarField,
        comm_result: PointCommitmentWithOpening<C>,
        result: Affine<P>,
        base: Affine<P>,
        comm_key: &PedersenCommitmentKey<Affine<C>>,
    ) -> Result<Self, Error> {
        let mut protocols = Vec::with_capacity(NUM_REPS);
        let twice_omega = omega.double();
        // Ensure that alpha is neither 0 nor omega (the scalar) nor 2*omega to avoid point doubling or points at infinity in point addition protocol
        let mut alpha = Vec::with_capacity(NUM_REPS);
        while alpha.len() < NUM_REPS {
            let alpha_i = P::ScalarField::rand(rng);
            if alpha_i.is_zero() || alpha_i == omega || alpha_i == twice_omega {
                continue;
            } else {
                alpha.push(alpha_i);
            }
        }

        // Randomness for the commitments to the points
        let beta_2 = (0..NUM_REPS)
            .map(|_| C::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let beta_3 = (0..NUM_REPS)
            .map(|_| C::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let beta_4 = (0..NUM_REPS)
            .map(|_| C::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let beta_5 = (0..NUM_REPS)
            .map(|_| C::ScalarField::rand(rng))
            .collect::<Vec<_>>();

        let base_table = WindowTable::new(NUM_REPS, base.into_group());
        // Points base * alpha_i
        let alpha_point = base_table.multiply_many(&alpha);
        // Point base * - omega
        let minus_omega_point = result.into_group().neg();
        // Points base * (alpha_i - omega)
        let alpha_minus_omega_point = cfg_iter!(alpha_point)
            .map(|a| minus_omega_point + a)
            .collect::<Vec<_>>();

        let alpha_point = <Affine<P> as AffineRepr>::Group::normalize_batch(&alpha_point);
        let alpha_minus_omega_point =
            <Affine<P> as AffineRepr>::Group::normalize_batch(&alpha_minus_omega_point);

        // Commit to base * alpha_i
        let comm_alpha_point_ = cfg_into_iter!(0..NUM_REPS)
            .map(|i| {
                PointCommitmentWithOpening::<C>::new_given_randomness::<P>(
                    &alpha_point[i],
                    beta_2[i],
                    beta_3[i],
                    comm_key,
                )
            })
            .collect::<Vec<_>>();
        let mut comm_alpha_point = Vec::with_capacity(NUM_REPS);
        for c in comm_alpha_point_ {
            comm_alpha_point.push(c?);
        }

        // Commit to base * (alpha_i - omega)
        let comm_alpha_minus_omega_point_ = cfg_into_iter!(0..NUM_REPS)
            .map(|i| {
                PointCommitmentWithOpening::<C>::new_given_randomness::<P>(
                    &alpha_minus_omega_point[i],
                    beta_4[i],
                    beta_5[i],
                    comm_key,
                )
            })
            .collect::<Vec<_>>();
        let mut comm_alpha_minus_omega_point = Vec::with_capacity(NUM_REPS);
        for c in comm_alpha_minus_omega_point_ {
            comm_alpha_minus_omega_point.push(c?);
        }

        // Following can be parallelized if PointAdditionProtocol and its sub-protocols accept randomness
        for i in 0..NUM_REPS {
            let add = PointAdditionProtocol::<P, C>::init(
                rng,
                comm_result.clone(),
                comm_alpha_minus_omega_point[0].clone(), // using index 0 because these are mutated below
                comm_alpha_point[0].clone(), // using index 0 because these are mutated below
                result,
                alpha_minus_omega_point[i],
                alpha_point[i],
                comm_key,
            )?;
            protocols.push(ScalarMultiplicationWCProtocolSingleRep {
                alpha: alpha.remove(0),
                comm_alpha_point: comm_alpha_point.remove(0),
                comm_alpha_minus_omega_point: comm_alpha_minus_omega_point.remove(0),
                add,
            });
        }
        Ok(Self {
            omega,
            sub_protocols: protocols,
        })
    }

    pub fn challenge_contribution<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        for i in 0..NUM_REPS {
            self.sub_protocols[i]
                .comm_alpha_point
                .comm
                .serialize_compressed(&mut writer)?;
            self.sub_protocols[i]
                .comm_alpha_minus_omega_point
                .comm
                .serialize_compressed(&mut writer)?;
            self.sub_protocols[i]
                .add
                .challenge_contribution(&mut writer)?;
        }
        Ok(())
    }

    pub fn gen_proof(self, challenge: &[u8]) -> ScalarMultiplicationWCProof<P, C, NUM_REPS> {
        // This assert should generally pass but can be avoided by enlarging the given challenge with an XOF
        assert!((challenge.len() * 8) >= NUM_REPS);
        let one = C::ScalarField::one();
        let minus_one = one.neg();
        let proofs = cfg_into_iter!(self.sub_protocols)
            .enumerate()
            .map(|(i, p)| {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                let c = (challenge[byte_idx] >> bit_idx) & 1;
                // If c = 0, send opening of point alpha * base else send opening of (alpha-omega) * base
                // If c = 0, the point addition protocol gets a challenge value of "-1" else it gets the value "1"
                if c == 0 {
                    ScalarMultiplicationWCProofSingleRep {
                        comm_alpha_point: p.comm_alpha_point.comm.clone(),
                        comm_alpha_minus_omega_point: p.comm_alpha_minus_omega_point.comm.clone(),
                        add: p.add.gen_proof(&minus_one),
                        z1: p.alpha,
                        z3: p.comm_alpha_point.r_x,
                        z4: p.comm_alpha_point.r_y,
                    }
                } else {
                    ScalarMultiplicationWCProofSingleRep {
                        comm_alpha_point: p.comm_alpha_point.comm.clone(),
                        comm_alpha_minus_omega_point: p.comm_alpha_minus_omega_point.comm.clone(),
                        add: p.add.gen_proof(&one),
                        z1: p.alpha - self.omega,
                        z3: p.comm_alpha_minus_omega_point.r_x,
                        z4: p.comm_alpha_minus_omega_point.r_y,
                    }
                }
            })
            .collect::<Vec<_>>();
        ScalarMultiplicationWCProof(proofs)
    }
}

impl<P: SWPoint, C: SWPoint, const NUM_REPS: usize> ScalarMultiplicationWCProof<P, C, NUM_REPS> {
    /// For verifying `base * scalar = result` where `comm_scalar` and `comm_result` are commitments to `scalar`
    /// and `result` respectively
    pub fn verify(
        &self,
        comm_result: &PointCommitment<C>,
        base: &Affine<P>,
        challenge: &[u8],
        comm_key: &PedersenCommitmentKey<Affine<C>>,
    ) -> Result<(), Error> {
        if self.0.len() != NUM_REPS {
            return Err(Error::InsufficientNumberOfRepetitions(
                self.0.len(),
                NUM_REPS,
            ));
        }
        if (challenge.len() * 8) < NUM_REPS {
            return Err(Error::InsufficientChallengeSize(
                challenge.len() * 8,
                NUM_REPS,
            ));
        }
        let base_table = WindowTable::new(NUM_REPS, base.into_group());
        let one = C::ScalarField::one();
        let minus_one = one.neg();
        // Following can be parallelized
        for i in 0..NUM_REPS {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let c = (challenge[byte_idx] >> bit_idx) & 1;
            // recompute the commitment to the point omegaP or (omega-alpha)P
            let p = (&base_table * &self.0[i].z1).into_affine();
            let p_comm = PointCommitmentWithOpening::new_given_randomness(
                &p,
                self.0[i].z3,
                self.0[i].z4,
                comm_key,
            )?;
            // If c = 0, expect opening of point alpha * base else expect opening of (alpha-omega) * base
            // If c = 0, the point addition protocol gets a challenge value of "-1" else it gets the value "1"
            if c == 0 {
                if p_comm.comm != self.0[i].comm_alpha_point {
                    return Err(Error::IncorrectPointOpeningAtIndex(i));
                }
                self.0[i].add.verify(
                    comm_result,
                    &self.0[i].comm_alpha_minus_omega_point,
                    &self.0[i].comm_alpha_point,
                    &minus_one,
                    comm_key,
                )?;
            } else {
                if p_comm.comm != self.0[i].comm_alpha_minus_omega_point {
                    return Err(Error::IncorrectPointOpeningAtIndex(i));
                }
                self.0[i].add.verify(
                    comm_result,
                    &self.0[i].comm_alpha_minus_omega_point,
                    &self.0[i].comm_alpha_point,
                    &one,
                    comm_key,
                )?;
            }
        }
        Ok(())
    }

    /// Same as `Self::verify` but delegated the scalar multiplication checks to `RandomizedMultChecker`
    pub fn verify_using_randomized_mult_checker(
        &self,
        comm_result: PointCommitment<C>,
        base: Affine<P>,
        challenge: &[u8],
        comm_key: PedersenCommitmentKey<Affine<C>>,
        rmc: &mut RandomizedMultChecker<Affine<C>>,
    ) -> Result<(), Error> {
        if self.0.len() != NUM_REPS {
            return Err(Error::InsufficientNumberOfRepetitions(
                self.0.len(),
                NUM_REPS,
            ));
        }
        if (challenge.len() * 8) < NUM_REPS {
            return Err(Error::InsufficientChallengeSize(
                challenge.len() * 8,
                NUM_REPS,
            ));
        }
        let base_table = WindowTable::new(NUM_REPS, base.into_group());
        let one = C::ScalarField::one();
        let minus_one = one.neg();
        for i in 0..NUM_REPS {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            let c = (challenge[byte_idx] >> bit_idx) & 1;
            let p = (&base_table * &self.0[i].z1).into_affine();
            let (p_x, p_y) = point_coords_as_scalar_field_elements::<P, C>(&p)?;
            if c == 0 {
                rmc.add_2(
                    comm_key.g,
                    &p_x,
                    comm_key.h,
                    &self.0[i].z3,
                    self.0[i].comm_alpha_point.x,
                );
                rmc.add_2(
                    comm_key.g,
                    &p_y,
                    comm_key.h,
                    &self.0[i].z4,
                    self.0[i].comm_alpha_point.y,
                );
                self.0[i].add.verify_using_randomized_mult_checker(
                    comm_result,
                    self.0[i].comm_alpha_minus_omega_point,
                    self.0[i].comm_alpha_point,
                    &minus_one,
                    comm_key,
                    rmc,
                )?;
            } else {
                rmc.add_2(
                    comm_key.g,
                    &p_x,
                    comm_key.h,
                    &self.0[i].z3,
                    self.0[i].comm_alpha_minus_omega_point.x,
                );
                rmc.add_2(
                    comm_key.g,
                    &p_y,
                    comm_key.h,
                    &self.0[i].z4,
                    self.0[i].comm_alpha_minus_omega_point.y,
                );
                self.0[i].add.verify_using_randomized_mult_checker(
                    comm_result,
                    self.0[i].comm_alpha_minus_omega_point,
                    self.0[i].comm_alpha_point,
                    &one,
                    comm_key,
                    rmc,
                )?;
            }
        }
        Ok(())
    }

    pub fn challenge_contribution<W: Write>(&self, mut writer: W) -> Result<(), Error> {
        for i in 0..NUM_REPS {
            self.0[i]
                .comm_alpha_point
                .serialize_compressed(&mut writer)?;
            self.0[i]
                .comm_alpha_minus_omega_point
                .serialize_compressed(&mut writer)?;
            self.0[i].add.challenge_contribution(&mut writer)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tom256::{Affine as tomAff, Config as tomConfig};
    use ark_secp256r1::{Affine as secpAff, Config as secpConfig, Fr as secpFr};
    use ark_std::UniformRand;
    use blake2::Blake2b512;
    use dock_crypto_utils::transcript::{new_merlin_transcript, Transcript};
    use rand_core::OsRng;
    use std::time::Instant;
    use test_utils::statistics::statistics;

    #[test]
    fn scalar_mult_wc() {
        let mut rng = OsRng::default();

        let comm_key = PedersenCommitmentKey::<tomAff>::new::<Blake2b512>(b"test2");

        let mut prov_time = vec![];
        let mut ver_time = vec![];
        let mut ver_rmc_time = vec![];
        let num_iters = 10;
        const NUM_REPS: usize = 128;
        for i in 0..num_iters {
            let base = secpAff::rand(&mut rng);
            let scalar = secpFr::rand(&mut rng);
            let result = (base * scalar).into_affine();

            let comm_result =
                PointCommitmentWithOpening::new(&mut rng, &result, &comm_key).unwrap();

            let start = Instant::now();
            let mut prover_transcript = new_merlin_transcript(b"test");
            prover_transcript.append(b"comm_key", &comm_key);
            prover_transcript.append(b"comm_result", &comm_result.comm);

            let protocol = ScalarMultiplicationWCProtocol::<secpConfig, tomConfig, NUM_REPS>::init(
                &mut rng,
                scalar,
                comm_result.clone(),
                result,
                base,
                &comm_key,
            )
            .unwrap();
            protocol
                .challenge_contribution(&mut prover_transcript)
                .unwrap();
            let mut challenge_prover = [0_u8; NUM_REPS / 8];
            prover_transcript.challenge_bytes(b"challenge", &mut challenge_prover);
            let proof = protocol.gen_proof(&challenge_prover);
            prov_time.push(start.elapsed());

            let start = Instant::now();
            let mut verifier_transcript = new_merlin_transcript(b"test");
            verifier_transcript.append(b"comm_key", &comm_key);
            verifier_transcript.append(b"comm_result", &comm_result.comm);
            proof
                .challenge_contribution(&mut verifier_transcript)
                .unwrap();

            let mut challenge_verifier = [0_u8; NUM_REPS / 8];
            verifier_transcript.challenge_bytes(b"challenge", &mut challenge_verifier);
            assert_eq!(challenge_prover, challenge_verifier);

            proof
                .verify(&comm_result.comm, &base, &challenge_verifier, &comm_key)
                .unwrap();
            ver_time.push(start.elapsed());

            let start = Instant::now();
            let mut verifier_transcript = new_merlin_transcript(b"test");
            verifier_transcript.append(b"comm_key", &comm_key);
            verifier_transcript.append(b"comm_result", &comm_result.comm);
            proof
                .challenge_contribution(&mut verifier_transcript)
                .unwrap();

            let mut challenge_verifier = [0_u8; NUM_REPS / 8];
            verifier_transcript.challenge_bytes(b"challenge", &mut challenge_verifier);
            assert_eq!(challenge_prover, challenge_verifier);

            let mut checker = RandomizedMultChecker::<tomAff>::new_using_rng(&mut rng);

            proof
                .verify_using_randomized_mult_checker(
                    comm_result.comm,
                    base,
                    &challenge_verifier,
                    comm_key,
                    &mut checker,
                )
                .unwrap();
            assert!(checker.verify());
            ver_rmc_time.push(start.elapsed());

            if i == 0 {
                println!("Proof size = {} bytes", proof.compressed_size());
            }
        }
        println!("For {num_iters} iterations");
        println!("Proving time: {:?}", statistics(prov_time));
        println!("Verifying time: {:?}", statistics(ver_time));
        println!(
            "Verifying time with randomized multiplication check: {:?}",
            statistics(ver_rmc_time)
        );
    }
}
