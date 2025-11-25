"use client";

import { Canvas } from "@react-three/fiber";
import { OrbitControls, Sphere, MeshDistortMaterial } from "@react-three/drei";
import { Suspense } from "react";

export default function Globe() {
  return (
    <div className="w-full h-[500px]">
      <Canvas camera={{ position: [0, 0, 4] }}>
        <Suspense fallback={null}>
          <ambientLight intensity={0.5} />
          <directionalLight position={[3, 3, 3]} intensity={1} />

          <Sphere args={[1, 64, 64]}>
            <MeshDistortMaterial
              color="#0ea5e9"
              distort={0.25}
              speed={2}
            />
          </Sphere>

          <OrbitControls enableZoom={false} autoRotate autoRotateSpeed={1.5} />
        </Suspense>
      </Canvas>
    </div>
  );
}