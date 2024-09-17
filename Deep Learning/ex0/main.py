from exercise0_material.src_to_implement import pattern

if __name__ == "__main__":
    checker = pattern.Checker(16, 2)
    checker.draw()
    checker.show()

    circle = pattern.Circle(80, 20, (40, 40))
    circle.draw()
    circle.show()

    spectrum = pattern.Spectrum(5)
    spectrum.draw()
    spectrum.show()
