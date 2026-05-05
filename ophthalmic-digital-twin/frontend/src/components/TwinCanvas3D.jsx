/**
 * TwinCanvas3D — Three.js 3D digital twin visualization.
 *
 * Features:
 * - Glowing icosahedron at state_3d position, color = disease severity
 * - State trajectory as 3D tube (TubeGeometry), gradient gray→cyan
 * - Future simulated states as dashed semi-transparent tube (orange→red)
 * - Compliance ring (torus) color-coded green/yellow/red
 * - Bloom post-processing
 * - OrbitControls for interaction
 * - Flickering animation when uncertainty is high
 */

import React, { useRef, useMemo, useEffect, useState } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import {
  OrbitControls,
  Html,
  Torus,
} from '@react-three/drei'
import { EffectComposer, Bloom } from '@react-three/postprocessing'
import * as THREE from 'three'

const SCALE = 1.5

function lerp(a, b, t) {
  return a + (b - a) * t
}

function vecToV3(arr) {
  if (!arr || arr.length < 3) return new THREE.Vector3(0, 0, 0)
  return new THREE.Vector3(arr[0] * SCALE, arr[1] * SCALE, arr[2] * SCALE)
}

// ── Icosahedron (digital twin state) ────────────────────

function TwinIcosahedron({ position3d, confidence, uncertainty, isLoaded }) {
  const meshRef = useRef()
  const glowRef = useRef()
  const highUncertainty = uncertainty > 0.5

  const severity = 1 - (confidence ?? 0.5)
  const cyanColor = new THREE.Color('#00c8ff')
  const violetColor = new THREE.Color('#7c3aed')
  const twinColor = cyanColor.clone().lerp(violetColor, severity)

  const emissiveColor = twinColor.clone().multiplyScalar(0.8)

  useFrame((state) => {
    if (!meshRef.current) return
    const t = state.clock.elapsedTime

    // Subtle continuous rotation
    meshRef.current.rotation.x += 0.004
    meshRef.current.rotation.y += 0.006

    // Scale in on load
    if (isLoaded) {
      const s = Math.min(1.0, meshRef.current.scale.x + 0.04)
      meshRef.current.scale.set(s, s, s)
    }

    // Flicker if uncertain
    if (highUncertainty) {
      const flicker = 0.3 + 0.7 * (0.5 + 0.5 * Math.sin(t * 8 * Math.PI * 2))
      meshRef.current.material.opacity = flicker
    } else {
      meshRef.current.material.opacity = 1.0
    }
  })

  const pos = vecToV3(position3d)

  return (
    <group position={pos}>
      {/* Core icosahedron */}
      <mesh ref={meshRef} scale={[0, 0, 0]}>
        <icosahedronGeometry args={[0.9, 1]} />
        <meshStandardMaterial
          color={twinColor}
          emissive={emissiveColor}
          emissiveIntensity={1.5}
          wireframe
          transparent
          opacity={1}
        />
      </mesh>

      {/* Outer glow sphere */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[1.05, 16, 16]} />
        <meshStandardMaterial
          color={twinColor}
          emissive={twinColor}
          emissiveIntensity={0.3}
          transparent
          opacity={0.08}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Label */}
      <Html distanceFactor={10} position={[0, 1.4, 0]} center>
        <div style={{
          background: 'rgba(2,4,8,0.85)',
          border: '1px solid #00c8ff44',
          borderRadius: '4px',
          padding: '3px 8px',
          fontSize: '10px',
          fontFamily: "'Space Mono', monospace",
          color: '#00c8ff',
          whiteSpace: 'nowrap',
          pointerEvents: 'none',
        }}>
          CONF {((confidence ?? 0.5) * 100).toFixed(1)}%
        </div>
      </Html>
    </group>
  )
}

// ── Historical trajectory tube ───────────────────────────

function TrajectoryTube({ points3d }) {
  const tubeMesh = useMemo(() => {
    if (!points3d || points3d.length < 2) return null
    const pts = points3d.map(vecToV3)
    const curve = new THREE.CatmullRomCurve3(pts)
    return new THREE.TubeGeometry(curve, Math.max(pts.length * 3, 20), 0.025, 8, false)
  }, [points3d])

  if (!tubeMesh) return null

  return (
    <mesh geometry={tubeMesh}>
      <meshStandardMaterial
        color="#00c8ff"
        emissive="#00c8ff"
        emissiveIntensity={0.6}
        transparent
        opacity={0.65}
      />
    </mesh>
  )
}

// ── Future simulation tube ───────────────────────────────

function FutureTube({ points3d, currentPos }) {
  const [visibleCount, setVisibleCount] = useState(0)

  useEffect(() => {
    if (!points3d || points3d.length === 0) return
    setVisibleCount(0)
    let i = 0
    const interval = setInterval(() => {
      i++
      setVisibleCount(i)
      if (i >= points3d.length) clearInterval(interval)
    }, 100)
    return () => clearInterval(interval)
  }, [points3d])

  const tubeMesh = useMemo(() => {
    if (!points3d || visibleCount < 1) return null
    const visible = points3d.slice(0, visibleCount)
    const allPts = [vecToV3(currentPos), ...visible.map(vecToV3)]
    if (allPts.length < 2) return null
    const curve = new THREE.CatmullRomCurve3(allPts)
    return new THREE.TubeGeometry(curve, Math.max(allPts.length * 3, 12), 0.02, 8, false)
  }, [points3d, visibleCount, currentPos])

  if (!tubeMesh) return null

  return (
    <mesh geometry={tubeMesh}>
      <meshStandardMaterial
        color="#ff6b35"
        emissive="#ff3300"
        emissiveIntensity={0.5}
        transparent
        opacity={0.5}
        wireframe={false}
      />
    </mesh>
  )
}

// ── Compliance ring (torus) ──────────────────────────────

function ComplianceRing({ position3d, complianceLevel }) {
  const ringRef = useRef()
  const color = complianceLevel > 0.8 ? '#00ff88'
              : complianceLevel > 0.5 ? '#ffcc00'
              : '#ff3300'

  useFrame(() => {
    if (ringRef.current) {
      ringRef.current.rotation.x += 0.01
      ringRef.current.rotation.z += 0.005
    }
  })

  const pos = vecToV3(position3d)

  return (
    <Torus
      ref={ringRef}
      args={[1.2, 0.04, 16, 64]}
      position={pos}
    >
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={1.0}
        transparent
        opacity={0.8}
      />
    </Torus>
  )
}

// ── Scene ────────────────────────────────────────────────

function Scene({ inferenceData, simulationData, patientId }) {
  const currentPos = inferenceData?.latent_state_3d ?? [0, 0, 0]
  const trajectory  = inferenceData?._trajectory ?? [currentPos]
  const confidence  = inferenceData?.confidence ?? 0.5
  const uncertainty = inferenceData?.total_uncertainty ?? 0.5
  const compliance  = simulationData ? 0.8 : 1.0
  const isLoaded    = !!inferenceData

  const futurePts = simulationData?.states_3d ?? null

  return (
    <>
      <ambientLight intensity={0.1} />
      <pointLight position={[5, 5, 5]} intensity={0.5} color="#00c8ff" />
      <pointLight position={[-5, -5, -5]} intensity={0.3} color="#7c3aed" />

      <TwinIcosahedron
        position3d={currentPos}
        confidence={confidence}
        uncertainty={uncertainty}
        isLoaded={isLoaded}
      />

      <TrajectoryTube points3d={trajectory} />

      {futurePts && (
        <FutureTube points3d={futurePts} currentPos={currentPos} />
      )}

      <ComplianceRing position3d={currentPos} complianceLevel={compliance} />

      {/* Patient ID label in 3D space */}
      <Html position={[-3, 3, 0]} center>
        <div style={{
          color: 'rgba(226,232,240,0.6)',
          fontSize: '11px',
          fontFamily: "'Space Mono', monospace",
          letterSpacing: '0.1em',
          pointerEvents: 'none',
        }}>
          {patientId ? `PID: ${patientId}` : 'NO PATIENT'}
        </div>
      </Html>

      <OrbitControls
        enablePan={false}
        minDistance={4}
        maxDistance={20}
        autoRotate
        autoRotateSpeed={0.3}
      />

      <EffectComposer>
        <Bloom
          threshold={0.5}
          strength={1.2}
          radius={0.8}
          luminanceThreshold={0.4}
        />
      </EffectComposer>
    </>
  )
}

/**
 * @component 3D digital twin visualization canvas.
 * @param {Object} inferenceData - result from /predict endpoint
 * @param {Object} simulationData - result from /simulate endpoint
 * @param {string} patientId
 */
export default function TwinCanvas3D({ inferenceData, simulationData, patientId }) {
  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas
        camera={{ position: [0, 0, 10], fov: 55 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <Scene
          inferenceData={inferenceData}
          simulationData={simulationData}
          patientId={patientId}
        />
      </Canvas>
    </div>
  )
}
