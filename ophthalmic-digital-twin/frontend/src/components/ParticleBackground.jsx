/**
 * ParticleBackground — full-screen Three.js animated particle field.
 * 2000 particles in a sphere formation with mouse parallax, Y-axis rotation,
 * and opacity pulsing via sin(time + index).
 */

import React, { useRef, useMemo, useEffect } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import * as THREE from 'three'

const PARTICLE_COUNT = 2000
const SPHERE_RADIUS = 8

function Particles() {
  const meshRef = useRef()
  const { camera } = useThree()
  const mouse = useRef({ x: 0, y: 0 })

  // Generate sphere positions and per-particle phase offset
  const { positions, phases, colors } = useMemo(() => {
    const pos = new Float32Array(PARTICLE_COUNT * 3)
    const ph = new Float32Array(PARTICLE_COUNT)
    const col = new Float32Array(PARTICLE_COUNT * 3)

    const cyan = new THREE.Color('#00c8ff')
    const violet = new THREE.Color('#7c3aed')

    for (let i = 0; i < PARTICLE_COUNT; i++) {
      // Uniform sphere distribution
      const theta = Math.random() * Math.PI * 2
      const phi = Math.acos(2 * Math.random() - 1)
      const r = SPHERE_RADIUS * Math.cbrt(Math.random())

      pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta)
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta)
      pos[i * 3 + 2] = r * Math.cos(phi)

      ph[i] = Math.random() * Math.PI * 2

      // Color lerp by depth (z-component normalized)
      const t = (pos[i * 3 + 2] + SPHERE_RADIUS) / (2 * SPHERE_RADIUS)
      const c = cyan.clone().lerp(violet, t)
      col[i * 3]     = c.r
      col[i * 3 + 1] = c.g
      col[i * 3 + 2] = c.b
    }
    return { positions: pos, phases: ph, colors: col }
  }, [])

  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return geo
  }, [positions, colors])

  const material = useMemo(() => new THREE.PointsMaterial({
    size: 0.045,
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    sizeAttenuation: true,
    depthWrite: false,
  }), [])

  // Track mouse for parallax
  useEffect(() => {
    const onMove = (e) => {
      mouse.current.x = (e.clientX / window.innerWidth - 0.5) * 2
      mouse.current.y = -(e.clientY / window.innerHeight - 0.5) * 2
    }
    window.addEventListener('mousemove', onMove)
    return () => window.removeEventListener('mousemove', onMove)
  }, [])

  useFrame((state) => {
    const t = state.clock.elapsedTime

    // Rotate entire point cloud slowly around Y axis
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.0003
    }

    // Mouse parallax: shift camera slightly
    camera.position.x += (mouse.current.x * 0.5 - camera.position.x) * 0.02
    camera.position.y += (mouse.current.y * 0.5 - camera.position.y) * 0.02
    camera.lookAt(0, 0, 0)

    // Pulse opacity via sin
    material.opacity = 0.4 + 0.35 * Math.sin(t * 0.8)
  })

  return <points ref={meshRef} geometry={geometry} material={material} />
}

/** @component Full-screen animated particle background using Three.js. */
export default function ParticleBackground() {
  return (
    <div style={{
      position: 'fixed',
      inset: 0,
      zIndex: 0,
      pointerEvents: 'none',
    }}>
      <Canvas
        camera={{ position: [0, 0, 12], fov: 60 }}
        gl={{ antialias: false, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <Particles />
      </Canvas>
    </div>
  )
}
