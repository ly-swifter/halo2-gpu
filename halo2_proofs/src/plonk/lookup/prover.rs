use super::super::{
    circuit::Expression, ChallengeBeta, ChallengeGamma, ChallengeTheta, ChallengeX, Error,
    ProvingKey,
};
use super::Argument;
use crate::plonk::evaluation::evaluate;
use crate::poly::kzg::commitment::KZGCommitmentScheme;
use crate::{
    arithmetic::{eval_polynomial, parallelize, CurveAffine, FieldExt},
    poly::{
        kzg::commitment::{commit_lagrange_gpu_sync, ParamsKZG,},
        commitment::{Blind, Params, CommitmentScheme,},
        Coeff, EvaluationDomain, ExtendedLagrangeCoeff, LagrangeCoeff, Polynomial, ProverQuery,
        Rotation, lagrange_to_coeff_gpu_sync,
    },
    transcript::{EncodedChallenge, TranscriptWrite},
    
};
use halo2curves::pairing::Engine;
use halo2curves::pasta::pallas::Scalar;
use crate::helpers::SerdeCurveAffine;

use crate::arithmetic::Group;


use group::{
    ff::{BatchInvert, Field},
    Curve,
};
use rand_core::RngCore;
use std::{any::TypeId, convert::TryInto, num::ParseIntError, ops::Index};
use std::{
    collections::BTreeMap,
    iter,
    ops::{Mul, MulAssign},
};

use zk_gpu::threadpool::{Waiter, Worker};
use std::sync::Arc;
use std::fmt::Debug;

#[derive(Debug, Clone)]
pub(in crate::plonk) struct Permuted<C: CurveAffine> {
    compressed_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_input_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_input_poly: Polynomial<C::Scalar, Coeff>,
    permuted_input_blind: Blind<C::Scalar>,
    compressed_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_table_expression: Polynomial<C::Scalar, LagrangeCoeff>,
    permuted_table_poly: Polynomial<C::Scalar, Coeff>,
    permuted_table_blind: Blind<C::Scalar>,
}

#[derive(Debug)]
pub(in crate::plonk) struct Committed<C: CurveAffine> {
    pub(in crate::plonk) permuted_input_poly: Polynomial<C::Scalar, Coeff>,
    permuted_input_blind: Blind<C::Scalar>,
    pub(in crate::plonk) permuted_table_poly: Polynomial<C::Scalar, Coeff>,
    permuted_table_blind: Blind<C::Scalar>,
    pub(in crate::plonk) product_poly: Polynomial<C::Scalar, Coeff>,
    product_blind: Blind<C::Scalar>,
}

pub(in crate::plonk) struct Evaluated<C: CurveAffine> {
    constructed: Committed<C>,
}

pub(in crate::plonk) fn commit_values_gpu<
    'a,
    'params: 'a,
    C,
    P: Params<'params, C>+Send+Sync+'static,
    F: FieldExt,
>(
    pool: &Worker,
    w_domain: Arc<EvaluationDomain<C::Scalar>>,
    w_params: Arc<P>,
    permuted: Permuted<C>,
) -> Waiter<(Permuted<C>, C, C)> 
where
    C: CurveAffine<ScalarExt = F>,
    C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
{
    let commit_values = |
        values: Polynomial<C::Scalar, LagrangeCoeff>, 
        w_domain: Arc<EvaluationDomain<C::Scalar>>,
        w_params: Arc<P>, 
        k: u32, | {
        let poly = lagrange_to_coeff_gpu_sync(w_domain.clone(), values.clone());
        //log::info!("lagrange_to_coeff in lookup's commit_permuted finished");

        let blind = Blind::default();
        let commitment= commit_lagrange_gpu_sync(w_params.get_g_lagrange(), values, k).unwrap().to_affine();
        (poly, blind, commitment)
    };

    //let g_lagrange = Arc::new((*params.get_g_lagrange()).clone());
    let k = w_params.k();

    let w_params_1 = w_params.clone();
    let w_params_2 = w_params.clone();

    let ret = pool.compute(move || 
    {
        // Commit to permuted input expression
        let (permuted_input_poly, permuted_input_blind, permuted_input_commitment) =
            commit_values(permuted.permuted_input_expression.clone(), w_domain.clone(), w_params_1, k);

        // Commit to permuted table expression
        let (permuted_table_poly, permuted_table_blind, permuted_table_commitment) =
            commit_values(permuted.permuted_table_expression.clone(), w_domain.clone(), w_params_2, k);
        (Permuted {
            compressed_input_expression: permuted.compressed_input_expression ,
            permuted_input_expression: permuted.permuted_input_expression,
            permuted_input_poly,
            permuted_input_blind,
            compressed_table_expression: permuted.compressed_table_expression,
            permuted_table_expression: permuted.permuted_table_expression,
            permuted_table_poly,
            permuted_table_blind,
        }, permuted_input_commitment, permuted_table_commitment)
    });
    ret
}

// pub(in crate::plonk) fn commit_values_gpu<
//     'params,
//     E: Engine<G1Affine=C, Scalar=G>+Debug,
//     Scheme: CommitmentScheme<Curve = C>,
//     C: CurveAffine<ScalarExt=G>,
//     G: Group+halo2curves::FieldExt,
// >(
//     pool: &Worker,
//     w_domain: Arc<EvaluationDomain<G>>,
//     params: &'params Scheme::ParamsProver,
//     permuted: Permuted<E::G1Affine>,
// ) -> Waiter<(Permuted<E::G1Affine>, E::G1Affine, E::G1Affine)> 
// where
//     E::G1Affine: SerdeCurveAffine,
//     E::G2Affine: SerdeCurveAffine,
// {
//     let mut commit_values = |
//         values: Polynomial<<E::G1Affine as CurveAffine>::ScalarExt, LagrangeCoeff>, 
//         w_domain: Arc<EvaluationDomain<G>>,
//         g_lagrange: Arc<Vec<C>>, 
//         k: u32, | {
//         let poly = lagrange_to_coeff_gpu_sync(w_domain.clone(), values);
//         //log::info!("lagrange_to_coeff in lookup's commit_permuted finished");

//         let blind :Blind<G> = Blind::default();
//         let commitment= commit_lagrange_gpu_sync::<E, Scheme, C, G>(g_lagrange, values, k).unwrap().to_affine();
//         (poly, blind, commitment)
//     };

//     let g_lagrange = Arc::new(params.get_g_lagrange());
//     let k = params.k();

//     let ret = pool.compute(move || 
//     {
//         // Commit to permuted input expression
//         let (permuted_input_poly, permuted_input_blind, permuted_input_commitment) =
//             commit_values(permuted.permuted_input_expression.clone(), w_domain.clone(), g_lagrange.clone(), k);

//         // Commit to permuted table expression
//         let (permuted_table_poly, permuted_table_blind, permuted_table_commitment) =
//             commit_values(permuted.permuted_table_expression.clone(), w_domain.clone(), g_lagrange.clone(), k);
//         (Permuted {
//             compressed_input_expression: permuted.compressed_input_expression ,
//             permuted_input_expression: permuted.permuted_input_expression,
//             permuted_input_poly,
//             permuted_input_blind,
//             compressed_table_expression: permuted.compressed_table_expression,
//             permuted_table_expression: permuted.permuted_table_expression,
//             permuted_table_poly,
//             permuted_table_blind,
//         }, permuted_input_commitment, permuted_table_commitment)
//     });
//     ret
// }

impl<F: FieldExt> Argument<F> {

    // pub(in crate::plonk) fn commit_permuted_gpu<
    //     'a,
    //     'params: 'a,
    //     En: Engine,
    //     C: En::G1Affine,
    //     P: Params<'params, En::G1Affine>+Send+Sync,
    //     E: EncodedChallenge<En::G1Affine>,
    //     R: RngCore,
    //     T: TranscriptWrite<En::G1Affine, E>+Copy+Send+Sync,
    // >(
    //     &self,
    //     pool: &Worker,
    //     pk: &ProvingKey<En::G1Affine>,
    //     params: &P,
    //     domain: &EvaluationDomain<En::G1Affine::Scalar>,
    //     theta: ChallengeTheta<En::G1Affine>,
    //     advice_values: &'a [Polynomial<En::Scalar, LagrangeCoeff>],
    //     fixed_values: &'a [Polynomial<En::Scalar, LagrangeCoeff>],
    //     instance_values: &'a [Polynomial<En::Scalar, LagrangeCoeff>],
    //     challenges: &'a [En::Scalar],
    //     mut rng: R,
    // ) -> Result<Waiter<Result<(Permuted<En::G1Affine>, En::G1Affine, En::G1Affine), Error>>, Error>
    // where
    //     En::G1Affine: SerdeCurveAffine<ScalarExt = F>,
    //     En::G2Affine: SerdeCurveAffine<ScalarExt = F>,
    //     C: CurveAffine<ScalarExt = F>,
    //     C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    // {
    //     let w_domain = Arc::new(pk.vk.domain.clone());
    //     let w_param = Arc::new(params.clone() as ParamsKZG<_>);
    //     // Closure to get values of expressions and compress them
    //     let compress_expressions = |expressions: &[Expression<En::Scalar>]| {
    //         let compressed_expression = expressions
    //             .iter()
    //             .map(|expression| {
    //                 domain.lagrange_from_vec(evaluate(
    //                     expression,
    //                     params.n() as usize,
    //                     1,
    //                     fixed_values,
    //                     advice_values,
    //                     instance_values,
    //                     challenges,
    //                 ))
    //             })
    //             .fold(domain.empty_lagrange(), |acc, expression| {
    //                 acc * *theta + &expression
    //             });
    //         compressed_expression
    //     };

    //     // Get values of input expressions involved in the lookup and compress them
    //     let compressed_input_expression = compress_expressions(self.input_expressions.as_slice());

    //     // Get values of table expressions involved in the lookup and compress them
    //     let compressed_table_expression = compress_expressions(&self.table_expressions);

    //     // Permute compressed (InputExpression, TableExpression) pair
    //     let (permuted_input_expression, permuted_table_expression) = permute_expression_pair(
    //         pk,
    //         params,
    //         domain,
    //         &mut rng,
    //         &compressed_input_expression,
    //         &compressed_table_expression,
    //     )?;
        
    //     let mut commit_values_gpu = |values: &Polynomial<En::Scalar, LagrangeCoeff>| {
    //         let poly = lagrange_to_coeff_gpu_sync(w_domain.clone(), values.clone());
    //         //log::info!("lagrange_to_coeff in lookup's commit_permuted finished");

    //         let blind = Blind::default();
    //         let commitment= commit_lagrange_gpu_sync(w_param.clone(), values.clone(), blind).unwrap().to_affine();
    //         (poly, blind, commitment)
    //     };

    //     let ret = pool.compute(move || 
    //     {
    //         // Commit to permuted input expression
    //         let (permuted_input_poly, permuted_input_blind, permuted_input_commitment) =
    //         commit_values_gpu(&permuted_input_expression);

    //         // Commit to permuted table expression
    //         let (permuted_table_poly, permuted_table_blind, permuted_table_commitment) =
    //             commit_values_gpu(&permuted_table_expression);
    //         Ok((Permuted {
    //             compressed_input_expression,
    //             permuted_input_expression,
    //             permuted_input_poly,
    //             permuted_input_blind,
    //             compressed_table_expression,
    //             permuted_table_expression,
    //             permuted_table_poly,
    //             permuted_table_blind,
    //         }, permuted_input_commitment, permuted_table_commitment))
    //     });
    //     Ok(ret)
        
    // }

    pub(in crate::plonk) fn write_point_permuted<
            C,
            E: EncodedChallenge<C>,
            T: TranscriptWrite<C, E>,
        >(
        &self,
        permuted_input_commitment: C,
        permuted_table_commitment: C,
        transcript: &mut T,)
    -> Result<(), Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        // Hash permuted input commitment
        transcript.write_point(permuted_input_commitment)?;

        // Hash permuted table commitment
        transcript.write_point(permuted_table_commitment)?;
        Ok(())
    }
    /// Given a Lookup with input expressions [A_0, A_1, ..., A_{m-1}] and table expressions
    /// [S_0, S_1, ..., S_{m-1}], this method
    /// - constructs A_compressed = \theta^{m-1} A_0 + theta^{m-2} A_1 + ... + \theta A_{m-2} + A_{m-1}
    ///   and S_compressed = \theta^{m-1} S_0 + theta^{m-2} S_1 + ... + \theta S_{m-2} + S_{m-1},
    /// - permutes A_compressed and S_compressed using permute_expression_pair() helper,
    ///   obtaining A' and S', and
    /// - constructs Permuted<C> struct using permuted_input_value = A', and
    ///   permuted_table_expression = S'.
    /// The Permuted<C> struct is used to update the Lookup, and is then returned.
    pub(in crate::plonk) fn commit_permuted<
        'a,
        'params: 'a,
        C,
        P: Params<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        &self,
        pk: &ProvingKey<C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &'a [C::Scalar],
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Permuted<C>, Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        // Closure to get values of expressions and compress them
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            let compressed_expression = expressions
                .iter()
                .map(|expression| {
                    pk.vk.domain.lagrange_from_vec(evaluate(
                        expression,
                        params.n() as usize,
                        1,
                        fixed_values,
                        advice_values,
                        instance_values,
                        challenges,
                    ))
                })
                .fold(domain.empty_lagrange(), |acc, expression| {
                    acc * *theta + &expression
                });
            compressed_expression
        };

        // Get values of input expressions involved in the lookup and compress them
        let compressed_input_expression = compress_expressions(&self.input_expressions);

        // Get values of table expressions involved in the lookup and compress them
        let compressed_table_expression = compress_expressions(&self.table_expressions);

        // Permute compressed (InputExpression, TableExpression) pair
        let (permuted_input_expression, permuted_table_expression) = permute_expression_pair(
            pk,
            params,
            domain,
            &mut rng,
            &compressed_input_expression,
            &compressed_table_expression,
        )?;

        // Closure to construct commitment to vector of values
        let mut commit_values = |values: &Polynomial<C::Scalar, LagrangeCoeff>| {
            let poly = pk.vk.domain.lagrange_to_coeff(values.clone());
            //log::info!("lagrange_to_coeff in lookup's commit_permuted finished");

            let blind = Blind(C::Scalar::random(&mut rng));
            let commitment = params.commit_lagrange(values, blind).to_affine();
            (poly, blind, commitment)
        };

        // Commit to permuted input expression
        let (permuted_input_poly, permuted_input_blind, permuted_input_commitment) =
            commit_values(&permuted_input_expression);

        // Commit to permuted table expression
        let (permuted_table_poly, permuted_table_blind, permuted_table_commitment) =
            commit_values(&permuted_table_expression);

        // Hash permuted input commitment
        transcript.write_point(permuted_input_commitment)?;

        // Hash permuted table commitment
        transcript.write_point(permuted_table_commitment)?;

        Ok(Permuted {
            compressed_input_expression,
            permuted_input_expression,
            permuted_input_poly,
            permuted_input_blind,
            compressed_table_expression,
            permuted_table_expression,
            permuted_table_poly,
            permuted_table_blind,
        })
    }

    pub(in crate::plonk) fn commit_permuted_partial<
        'a,
        'params: 'a,
        C,
        P: Params<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        &self,
        pk: &ProvingKey<C>,
        params: &P,
        domain: &EvaluationDomain<C::Scalar>,
        theta: ChallengeTheta<C>,
        advice_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        fixed_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        instance_values: &'a [Polynomial<C::Scalar, LagrangeCoeff>],
        challenges: &'a [C::Scalar],
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Permuted<C>, Error>
    where
        C: CurveAffine<ScalarExt = F>,
        C::Curve: Mul<F, Output = C::Curve> + MulAssign<F>,
    {
        // Closure to get values of expressions and compress them
        let compress_expressions = |expressions: &[Expression<C::Scalar>]| {
            let compressed_expression = expressions
                .iter()
                .map(|expression| {
                    pk.vk.domain.lagrange_from_vec(evaluate(
                        expression,
                        params.n() as usize,
                        1,
                        fixed_values,
                        advice_values,
                        instance_values,
                        challenges,
                    ))
                })
                .fold(domain.empty_lagrange(), |acc, expression| {
                    acc * *theta + &expression
                });
            compressed_expression
        };

        // Get values of input expressions involved in the lookup and compress them
        let compressed_input_expression = compress_expressions(&self.input_expressions);

        // Get values of table expressions involved in the lookup and compress them
        let compressed_table_expression = compress_expressions(&self.table_expressions);

        // Permute compressed (InputExpression, TableExpression) pair
        let (permuted_input_expression, permuted_table_expression) = permute_expression_pair(
            pk,
            params,
            domain,
            &mut rng,
            &compressed_input_expression,
            &compressed_table_expression,
        )?;

        Ok(Permuted {
            compressed_input_expression,
            permuted_input_expression,
            permuted_input_poly: domain.empty_coeff(),
            permuted_input_blind: Blind(C::Scalar::random(&mut rng)),
            compressed_table_expression,
            permuted_table_expression,
            permuted_table_poly: domain.empty_coeff(),
            permuted_table_blind: Blind(C::Scalar::random(&mut rng)),
        })
    }
}


impl<C: CurveAffine> Permuted<C> {
    /// Given a Lookup with input expressions, table expressions, and the permuted
    /// input expression and permuted table expression, this method constructs the
    /// grand product polynomial over the lookup. The grand product polynomial
    /// is used to populate the Product<C> struct. The Product<C> struct is
    /// added to the Lookup and finally returned by the method.
    pub(in crate::plonk) fn commit_product<
        'params,
        P: Params<'params, C>,
        E: EncodedChallenge<C>,
        R: RngCore,
        T: TranscriptWrite<C, E>,
    >(
        self,
        pk: &ProvingKey<C>,
        params: &P,
        beta: ChallengeBeta<C>,
        gamma: ChallengeGamma<C>,
        mut rng: R,
        transcript: &mut T,
    ) -> Result<Committed<C>, Error> {
        let blinding_factors = pk.vk.cs.blinding_factors();
        // Goal is to compute the products of fractions
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where a_j(X) is the jth input expression in this lookup,
        // where a'(X) is the compression of the permuted input expressions,
        // s_j(X) is the jth table expression in this lookup,
        // s'(X) is the compression of the permuted table expressions,
        // and i is the ith row of the expression.
        let mut lookup_product = vec![C::Scalar::zero(); params.n() as usize];
        // Denominator uses the permuted input expression and permuted table expression
        parallelize(&mut lookup_product, |lookup_product, start| {
            for ((lookup_product, permuted_input_value), permuted_table_value) in lookup_product
                .iter_mut()
                .zip(self.permuted_input_expression[start..].iter())
                .zip(self.permuted_table_expression[start..].iter())
            {
                *lookup_product = (*beta + permuted_input_value) * &(*gamma + permuted_table_value);
            }
        });

        // Batch invert to obtain the denominators for the lookup product
        // polynomials
        lookup_product.iter_mut().batch_invert();

        // Finish the computation of the entire fraction by computing the numerators
        // (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        // * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        parallelize(&mut lookup_product, |product, start| {
            for (i, product) in product.iter_mut().enumerate() {
                let i = i + start;

                *product *= &(self.compressed_input_expression[i] + &*beta);
                *product *= &(self.compressed_table_expression[i] + &*gamma);
            }
        });

        // The product vector is a vector of products of fractions of the form
        //
        // Numerator: (\theta^{m-1} a_0(\omega^i) + \theta^{m-2} a_1(\omega^i) + ... + \theta a_{m-2}(\omega^i) + a_{m-1}(\omega^i) + \beta)
        //            * (\theta^{m-1} s_0(\omega^i) + \theta^{m-2} s_1(\omega^i) + ... + \theta s_{m-2}(\omega^i) + s_{m-1}(\omega^i) + \gamma)
        // Denominator: (a'(\omega^i) + \beta) (s'(\omega^i) + \gamma)
        //
        // where there are m input expressions and m table expressions,
        // a_j(\omega^i) is the jth input expression in this lookup,
        // a'j(\omega^i) is the permuted input expression,
        // s_j(\omega^i) is the jth table expression in this lookup,
        // s'(\omega^i) is the permuted table expression,
        // and i is the ith row of the expression.

        // Compute the evaluations of the lookup product polynomial
        // over our domain, starting with z[0] = 1
        let z = iter::once(C::Scalar::one())
            .chain(lookup_product)
            .scan(C::Scalar::one(), |state, cur| {
                *state *= &cur;
                Some(*state)
            })
            // Take all rows including the "last" row which should
            // be a boolean (and ideally 1, else soundness is broken)
            .take(params.n() as usize - blinding_factors)
            // Chain random blinding factors.
            .chain((0..blinding_factors).map(|_| C::Scalar::random(&mut rng)))
            .collect::<Vec<_>>();
        assert_eq!(z.len(), params.n() as usize);
        let z = pk.vk.domain.lagrange_from_vec(z);

        #[cfg(feature = "sanity-checks")]
        // This test works only with intermediate representations in this method.
        // It can be used for debugging purposes.
        {
            // While in Lagrange basis, check that product is correctly constructed
            let u = (params.n() as usize) - (blinding_factors + 1);

            // l_0(X) * (1 - z(X)) = 0
            assert_eq!(z[0], C::Scalar::one());

            // z(\omega X) (a'(X) + \beta) (s'(X) + \gamma)
            // - z(X) (\theta^{m-1} a_0(X) + ... + a_{m-1}(X) + \beta) (\theta^{m-1} s_0(X) + ... + s_{m-1}(X) + \gamma)
            for i in 0..u {
                let mut left = z[i + 1];
                let permuted_input_value = &self.permuted_input_expression[i];

                let permuted_table_value = &self.permuted_table_expression[i];

                left *= &(*beta + permuted_input_value);
                left *= &(*gamma + permuted_table_value);

                let mut right = z[i];
                let mut input_term = self.compressed_input_expression[i];
                let mut table_term = self.compressed_table_expression[i];

                input_term += &(*beta);
                table_term += &(*gamma);
                right *= &(input_term * &table_term);

                assert_eq!(left, right);
            }

            // l_last(X) * (z(X)^2 - z(X)) = 0
            // Assertion will fail only when soundness is broken, in which
            // case this z[u] value will be zero. (bad!)
            assert_eq!(z[u], C::Scalar::one());
        }

        let product_blind = Blind(C::Scalar::random(rng));
        let product_commitment = params.commit_lagrange(&z, product_blind).to_affine();
        let z = pk.vk.domain.lagrange_to_coeff(z);
        //log::info!("lagrange_to_coeff in lookup's commit_product finished");

        // Hash product commitment
        transcript.write_point(product_commitment)?;

        Ok(Committed::<C> {
            permuted_input_poly: self.permuted_input_poly,
            permuted_input_blind: self.permuted_input_blind,
            permuted_table_poly: self.permuted_table_poly,
            permuted_table_blind: self.permuted_table_blind,
            product_poly: z,
            product_blind,
        })
    }
}

impl<C: CurveAffine> Committed<C> {
    pub(in crate::plonk) fn evaluate<E: EncodedChallenge<C>, T: TranscriptWrite<C, E>>(
        self,
        pk: &ProvingKey<C>,
        x: ChallengeX<C>,
        transcript: &mut T,
    ) -> Result<Evaluated<C>, Error> {
        let domain = &pk.vk.domain;
        let x_inv = domain.rotate_omega(*x, Rotation::prev());
        let x_next = domain.rotate_omega(*x, Rotation::next());

        let product_eval = eval_polynomial(&self.product_poly, *x);
        let product_next_eval = eval_polynomial(&self.product_poly, x_next);
        let permuted_input_eval = eval_polynomial(&self.permuted_input_poly, *x);
        let permuted_input_inv_eval = eval_polynomial(&self.permuted_input_poly, x_inv);
        let permuted_table_eval = eval_polynomial(&self.permuted_table_poly, *x);

        // Hash each advice evaluation
        for eval in iter::empty()
            .chain(Some(product_eval))
            .chain(Some(product_next_eval))
            .chain(Some(permuted_input_eval))
            .chain(Some(permuted_input_inv_eval))
            .chain(Some(permuted_table_eval))
        {
            transcript.write_scalar(eval)?;
        }

        Ok(Evaluated { constructed: self })
    }
}

impl<C: CurveAffine> Evaluated<C> {
    pub(in crate::plonk) fn open<'a>(
        &'a self,
        pk: &'a ProvingKey<C>,
        x: ChallengeX<C>,
    ) -> impl Iterator<Item = ProverQuery<'a, C>> + Clone {
        let x_inv = pk.vk.domain.rotate_omega(*x, Rotation::prev());
        let x_next = pk.vk.domain.rotate_omega(*x, Rotation::next());

        iter::empty()
            // Open lookup product commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.product_poly,
                blind: self.constructed.product_blind,
            }))
            // Open lookup input commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.permuted_input_poly,
                blind: self.constructed.permuted_input_blind,
            }))
            // Open lookup table commitments at x
            .chain(Some(ProverQuery {
                point: *x,
                poly: &self.constructed.permuted_table_poly,
                blind: self.constructed.permuted_table_blind,
            }))
            // Open lookup input commitments at x_inv
            .chain(Some(ProverQuery {
                point: x_inv,
                poly: &self.constructed.permuted_input_poly,
                blind: self.constructed.permuted_input_blind,
            }))
            // Open lookup product commitments at x_next
            .chain(Some(ProverQuery {
                point: x_next,
                poly: &self.constructed.product_poly,
                blind: self.constructed.product_blind,
            }))
    }
}

type ExpressionPair<F> = (Polynomial<F, LagrangeCoeff>, Polynomial<F, LagrangeCoeff>);

/// Given a vector of input values A and a vector of table values S,
/// this method permutes A and S to produce A' and S', such that:
/// - like values in A' are vertically adjacent to each other; and
/// - the first row in a sequence of like values in A' is the row
///   that has the corresponding value in S'.
/// This method returns (A', S') if no errors are encountered.
fn permute_expression_pair<'params, C: CurveAffine, P: Params<'params, C>, R: RngCore>(
    pk: &ProvingKey<C>,
    params: &P,
    domain: &EvaluationDomain<C::Scalar>,
    mut rng: R,
    input_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
    table_expression: &Polynomial<C::Scalar, LagrangeCoeff>,
) -> Result<ExpressionPair<C::Scalar>, Error> {
    //log::info!("Start permute_expression_pair!");

    let blinding_factors = pk.vk.cs.blinding_factors();
    let usable_rows = params.n() as usize - (blinding_factors + 1);

    let mut permuted_input_expression: Vec<C::Scalar> = input_expression.to_vec();
    permuted_input_expression.truncate(usable_rows);

    // Sort input lookup expression values
    permuted_input_expression.sort();

    // A BTreeMap of each unique element in the table expression and its count
    let mut leftover_table_map: BTreeMap<C::Scalar, u32> = table_expression
        .iter()
        .take(usable_rows)
        .fold(BTreeMap::new(), |mut acc, coeff| {
            *acc.entry(*coeff).or_insert(0) += 1;
            acc
        });
    let mut permuted_table_coeffs = vec![C::Scalar::zero(); usable_rows];

    let mut repeated_input_rows = permuted_input_expression
        .iter()
        .zip(permuted_table_coeffs.iter_mut())
        .enumerate()
        .filter_map(|(row, (input_value, table_value))| {
            // If this is the first occurrence of `input_value` in the input expression
            if row == 0 || *input_value != permuted_input_expression[row - 1] {
                *table_value = *input_value;
                // Remove one instance of input_value from leftover_table_map
                if let Some(count) = leftover_table_map.get_mut(input_value) {
                    assert!(*count > 0);
                    *count -= 1;
                    None
                } else {
                    // Return error if input_value not found
                    Some(Err(Error::ConstraintSystemFailure))
                }
            // If input value is repeated
            } else {
                Some(Ok(row))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Populate permuted table at unfilled rows with leftover table elements
    for (coeff, count) in leftover_table_map.iter() {
        for _ in 0..*count {
            permuted_table_coeffs[repeated_input_rows.pop().unwrap() as usize] = *coeff;
        }
    }
    assert!(repeated_input_rows.is_empty());

    permuted_input_expression
        .extend((0..(blinding_factors + 1)).map(|_| C::Scalar::random(&mut rng)));
    permuted_table_coeffs.extend((0..(blinding_factors + 1)).map(|_| C::Scalar::random(&mut rng)));
    assert_eq!(permuted_input_expression.len(), params.n() as usize);
    assert_eq!(permuted_table_coeffs.len(), params.n() as usize);

    #[cfg(feature = "sanity-checks")]
    {
        let mut last = None;
        for (a, b) in permuted_input_expression
            .iter()
            .zip(permuted_table_coeffs.iter())
            .take(usable_rows)
        {
            if *a != *b {
                assert_eq!(*a, last.unwrap());
            }
            last = Some(*a);
        }
    }
    //log::info!("Finish permute_expression_pair!");

    Ok((
        domain.lagrange_from_vec(permuted_input_expression),
        domain.lagrange_from_vec(permuted_table_coeffs),
    ))
}
